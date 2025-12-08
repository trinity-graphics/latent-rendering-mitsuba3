#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/filesystem.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/string.h>
#include <mitsuba/render/film.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/imageblock.h>

#include <mutex>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class LatFilm final : public Film<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Film, m_size, m_crop_size, m_crop_offset, m_sample_border,
                   m_filter, m_flags)
    MI_IMPORT_TYPES(ImageBlock)

    static constexpr size_t lat_ch = Spectrum::Size;

    LatFilm(const Properties &props) : Base(props) {
        std::string component_format = string::to_lower(
            props.get<std::string_view>("component_format", "float16"));

        m_flags = +FilmFlags::Empty;
        m_file_format = Bitmap::FileFormat::OpenEXR;
        m_pixel_format = Bitmap::PixelFormat::MultiChannel;

        if (component_format == "float16")
            m_component_format = Struct::Type::Float16;
        else if (component_format == "float32")
            m_component_format = Struct::Type::Float32;
        else if (component_format == "uint32")
            m_component_format = Struct::Type::UInt32;
        else
            Throw("The \"component_format\" parameter must either be "
                  "equal to \"float16\", \"float32\", or \"uint32\"."
                  " Found %s instead.", component_format);

        m_compensate = props.get<bool>("compensate", false);

        props.mark_queried("banner"); // no banner in Mitsuba 3
    }

    size_t base_channels_count() const override {
        return lat_ch;
    }

    size_t prepare(const std::vector<std::string> &aovs) override {
        size_t base_channels = lat_ch + 1;

        std::vector<std::string> channels(base_channels + aovs.size());

        // Add basic RGBAW channels to the film
        const char *base_channel_names = "ch00ch01ch02ch03ch04ch05ch06ch07ch08ch09ch10ch11ch12ch13ch14ch15WWWW";

        for (size_t i = 0; i < base_channels; ++i)
            channels[i] = std::string(base_channel_names + 4*i, 4);

        for (size_t i = 0; i < aovs.size(); ++i)
            channels[base_channels + i] = aovs[i];

        /* locked */ {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_storage = new ImageBlock(m_crop_size, m_crop_offset,
                                       (uint32_t) channels.size());
            m_channels = channels;
        }

        std::sort(channels.begin(), channels.end());
        auto it = std::unique(channels.begin(), channels.end());
        if (it != channels.end())
            Throw("Film::prepare(): duplicate channel name \"%s\"", *it);

        return m_channels.size();
    }

    ref<ImageBlock> create_block(const ScalarVector2u &size, bool normalize,
                                 bool border) override {
        bool default_config = dr::all(size == ScalarVector2u(0));

        return new ImageBlock(default_config ? m_crop_size : size,
                              default_config ? m_crop_offset : ScalarPoint2u(0),
                              (uint32_t) m_channels.size(), m_filter.get(),
                              border /* border */,
                              normalize /* normalize */,
                              dr::is_jit_v<Float> /* coalesce */,
                              m_compensate /* compensate */,
                              false /* warn_negative */,
                              false /* warn_invalid */);
    }

    void put_block(const ImageBlock *block) override {
        Assert(m_storage != nullptr);
        std::lock_guard<std::mutex> lock(m_mutex);
        m_storage->put_block(block);
    }

    void clear() override {
        if (m_storage)
            m_storage->clear();
    }

    TensorXf develop(bool raw = false) const override {
        if (!m_storage)
            Throw("No storage allocated, was prepare() called first?");

        if (raw) {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_storage->tensor();
        }

        if constexpr (dr::is_jit_v<Float>) {
            Float data;
            uint32_t source_ch;
            uint32_t pixel_count;
            ScalarVector2i size;

            /* locked */ {
                std::lock_guard<std::mutex> lock(m_mutex);
                data        = m_storage->tensor().array();
                size        = m_storage->size();
                source_ch   = (uint32_t) m_storage->channel_count();
                pixel_count = dr::prod(m_storage->size());
            }

            /* The following code develops weighted image block data into
               an output image of the desired configuration, while using
               a minimal number of JIT kernel launches. */

            // Number of arbitrary output variables (AOVs)
            uint32_t base_ch = lat_ch + 1,
                     aovs    = source_ch - base_ch;

            /// Number of desired color components
            uint32_t color_ch = lat_ch;

            // Number of channels of the target tensor
            uint32_t target_ch = color_ch + aovs;

            // Index vectors referencing pixels & channels of the output image
            UInt32 idx         = dr::arange<UInt32>(pixel_count * target_ch),
                   pixel_idx   = idx / target_ch,
                   channel_idx = dr::fmadd(pixel_idx, uint32_t(-(int) target_ch), idx);

            /* Index vectors referencing source pixels/weights as follows:
                 values_idx = R1, G1, B1, R2, G2, B2 (for RGB output)
                 weight_idx = W1, W1, W1, W2, W2, W2 */
            UInt32 values_idx = dr::fmadd(pixel_idx, source_ch, channel_idx),
                   weight_idx = dr::fmadd(pixel_idx, source_ch, base_ch - 1);

            // If AOVs are desired, their indices in 'values_idx' must be shifted
            if (aovs) {
                // Index of first AOV channel in output image
                uint32_t first_aov = color_ch;
                values_idx[channel_idx >= first_aov] += base_ch - first_aov;
            }

            Mask value_mask = true;

            // Gather the pixel values from the image data buffer
            Float weight = dr::gather<Float>(data, weight_idx),
                  values = dr::gather<Float>(data, values_idx, value_mask);

            // Perform the weight division unless the weight is zero
            values /= dr::select(weight == 0.f, 1.f, weight);

            size_t shape[3] = { (size_t) size.y(), (size_t) size.x(),
                                target_ch };

            return TensorXf(values, 3, shape);
        } else {
            ref<Bitmap> source = bitmap();
            ScalarVector2i size = source->size();
            size_t width = source->channel_count() * dr::prod(size);
            auto data = dr::load<DynamicBuffer<ScalarFloat>>(source->data(), width);

            size_t shape[3] = { (size_t) source->height(),
                                (size_t) source->width(),
                                source->channel_count() };

            return TensorXf(data, 3, shape);
        }
    }

    ref<Bitmap> bitmap(bool raw = false) const override {
        if (!m_storage)
            Throw("No storage allocated, was prepare() called first?");

        std::lock_guard<std::mutex> lock(m_mutex);
        auto &&storage = dr::migrate(m_storage->tensor().array(), AllocType::Host);

        if constexpr (dr::is_jit_v<Float>)
            dr::sync_thread();

        uint32_t base_ch = lat_ch + 1;
        bool has_aovs  = m_channels.size() != base_ch;

        ref<Bitmap> source = new Bitmap(
            Bitmap::PixelFormat::MultiChannel, 
            struct_type_v<ScalarFloat>, m_storage->size(),
            m_storage->channel_count(), m_channels, (uint8_t *) storage.data());

        if (raw)
            return source;

        uint32_t img_ch = lat_ch;
        uint32_t aovs_channel = has_aovs ? img_ch : 0;
        uint32_t target_ch =
            (uint32_t) m_storage->channel_count() - base_ch + aovs_channel;

        ref<Bitmap> target = new Bitmap(
            Bitmap::PixelFormat::MultiChannel,
            struct_type_v<ScalarFloat>, m_storage->size(),
            has_aovs ? target_ch : img_ch);
        
        if (has_aovs) {
            source->struct_()->operator[](base_ch - 1).flags |=
                +Struct::Flags::Weight;

            for (size_t i = 0; i < target_ch; ++i) {
                Struct::Field &dest_field = target->struct_()->operator[](i);
                dest_field.name = m_channels[base_ch + i - aovs_channel];
            }
        }

        source->convert(target);

        return target;
    }

    void write(const fs::path &path) const override {
        fs::path filename = path;
        std::string proper_extension = ".exr";

        std::string extension = string::to_lower(filename.extension().string());
        if (extension != proper_extension)
            filename.replace_extension(proper_extension);

        #if !defined(_WIN32)
            Log(Info, "\U00002714  Developing \"%s\" ..", filename.string());
        #else
            Log(Info, "Developing \"%s\" ..", filename.string());
        #endif

        ref<Bitmap> source = bitmap();
        if (m_component_format != struct_type_v<ScalarFloat>) {
            // Mismatch between the current format and the one expected by the film
            // Conversion is necessary before saving to disk
            std::vector<std::string> channel_names;
            for (size_t i = 0; i < source->channel_count(); i++)
                channel_names.push_back(source->struct_()->operator[](i).name);
            ref<Bitmap> target = new Bitmap(
                source->pixel_format(),
                m_component_format,
                source->size(),
                source->channel_count(),
                channel_names);
            source->convert(target);

            target->write(filename, m_file_format);
        } else {
            source->write(filename, m_file_format);
        }
    }

    void schedule_storage() override {
        dr::schedule(m_storage->tensor());
    };

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "LatFilm[" << std::endl
            << "  size = " << m_size << "," << std::endl
            << "  crop_size = " << m_crop_size << "," << std::endl
            << "  crop_offset = " << m_crop_offset << "," << std::endl
            << "  sample_border = " << m_sample_border << "," << std::endl
            << "  compensate = " << m_compensate << "," << std::endl
            << "  filter = " << m_filter << "," << std::endl
            << "  file_format = " << m_file_format << "," << std::endl
            << "  pixel_format = " << m_pixel_format << "," << std::endl
            << "  component_format = " << m_component_format << "," << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(LatFilm)
protected:
    Bitmap::FileFormat m_file_format;
    Bitmap::PixelFormat m_pixel_format;
    Struct::Type m_component_format;
    bool m_compensate;
    ref<ImageBlock> m_storage;
    mutable std::mutex m_mutex;
    std::vector<std::string> m_channels;

    MI_TRAVERSE_CB(Base, m_storage)

};
MI_EXPORT_PLUGIN(LatFilm)
NAMESPACE_END(mitsuba)