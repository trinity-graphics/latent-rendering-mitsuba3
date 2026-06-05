#include <mitsuba/render/texture.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/core/properties.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class LatentTexture final : public Texture<Float, Spectrum> {
public:
    MI_IMPORT_TYPES(Texture)

    LatentTexture(const Properties &props) : Texture(props) {
        // frozen bitmap
        m_texture = props.get_texture<Texture>("texture", 0.f);
        // grayscale shifting parameters
        m_scale  = props.get_texture<Texture>("tex_scale", 1.f);
        m_offset = props.get_texture<Texture>("tex_offset", 0.f);
    }

    void traverse(TraversalCallback *cb) override {
        cb->put("texture", m_texture, ParamFlags::NonDifferentiable);
        cb->put("tex_scale", m_scale, ParamFlags::Differentiable);
        cb->put("tex_offset", m_offset, ParamFlags::Differentiable);
    }

    UnpolarizedSpectrum eval(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        return m_texture->eval(si, active) * m_scale->eval(si, active) + m_offset->eval(si, active);
        // This yields NaN values for some reason, have to use the above.
        // return dr::fmadd(
        //     m_texture->eval(si, active),
        //     m_scale->eval(si, active),
        //     m_offset->eval(si, active)
        // );
    }

    Float eval_1(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        return m_texture->eval_1(si, active) * m_scale->eval_1(si, active) + m_offset->eval_1(si, active);
    }

    Float mean() const override { 
        return m_texture->mean();
        // return m_texture->mean() * m_scale->mean(si, active) + m_offset->mean(si, active);
    }

    bool is_spatially_varying() const override { return true; }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "LatentTexture[" << std::endl
            << "  texture = " << string::indent(m_texture) << "," << std::endl
            << "  tex_scale  = " << string::indent(m_scale)  << std::endl
            << "  tex_offset  = " << string::indent(m_offset)  << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(LatentTexture)
protected:
    ref<Texture> m_texture;
    ref<Texture> m_scale;
    ref<Texture> m_offset;

    MI_TRAVERSE_CB(Texture, m_texture, m_scale, m_offset)
};

MI_EXPORT_PLUGIN(LatentTexture)
NAMESPACE_END(mitsuba)
