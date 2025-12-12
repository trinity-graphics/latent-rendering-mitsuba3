#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/render/srgb.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class LatentReflectanceSpectrum final : public Texture<Float, Spectrum> {
public:
    MI_IMPORT_TYPES(Texture)

    LatentReflectanceSpectrum(const Properties &props) : Texture(props) {
        if constexpr (!is_latent_v<Spectrum>)
            Throw("LatentReflectanceSpectrum can only be used with latent Mitsuba variants!");
        
        // props.get<ScalarColor1f>("color");
        m_value = props.get<Color<Float, ChannelCount>>("color");
        // ScalarColor3f color = props.get<ScalarColor3f>("color");

        dr::make_opaque(m_value);
    }

    void traverse(TraversalCallback *cb) override {
        cb->put("value", m_value, ParamFlags::Differentiable);
    }

    void parameters_changed(const std::vector<std::string> &/*keys*/ = {}) override {
        dr::make_opaque(m_value);
    }

    UnpolarizedSpectrum eval(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        return m_value;
    }

    Color3f eval_3(const SurfaceInteraction3f &/*si*/, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        return Color3f(m_value[0]);
    }

    Float eval_1(const SurfaceInteraction3f & /*it*/, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        return mean();
    }

    std::pair<Wavelength, UnpolarizedSpectrum>
    sample_spectrum(const SurfaceInteraction3f &_si,
                    const Wavelength &sample,
                    Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureSample, active);
        DRJIT_MARK_USED(sample);
        UnpolarizedSpectrum value = eval(_si, active);
        return { dr::empty<Wavelength>(), value };
    }

    Float mean() const override {
        return dr::mean(dr::mean(m_value));
    }

    ScalarFloat max() const override {
        return dr::max_nested(m_value);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "LatentReflectanceSpectrum[" << std::endl
            << "  value = " << string::indent(m_value) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(LatentReflectanceSpectrum)
protected:
    /**
     * Depending on the compiled variant, this plugin either stores coefficients
     * for a spectral upsampling model, or a plain RGB/monochromatic value.
     */
    static constexpr size_t ChannelCount = Spectrum::Size;

    Color<Float, ChannelCount> m_value;

    MI_TRAVERSE_CB(Texture, m_value)
};

MI_EXPORT_PLUGIN(LatentReflectanceSpectrum)
NAMESPACE_END(mitsuba)