#include <mitsuba/render/texture.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/core/properties.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class LatentTexture final : public Texture<Float, Spectrum> {
public:
    MI_IMPORT_TYPES(Texture)

    LatentTexture(const Properties &props) : Texture(props) {
        m_texture = props.get_texture<Texture>("texture", 0.f);  // frozen bitmap
        m_color  = props.get_texture<Texture>("tex_color", 1.f); // learnable, default = identity
    }

    void traverse(TraversalCallback *cb) override {
        cb->put("texture", m_texture, ParamFlags::NonDifferentiable);
        cb->put("tex_color", m_color, ParamFlags::Differentiable);
    }

    UnpolarizedSpectrum eval(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        return m_texture->eval(si, active) * m_color->eval(si, active);
    }

    Float eval_1(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        return m_texture->eval_1(si, active);
    }

    Float mean() const override { return m_texture->mean(); }

    bool is_spatially_varying() const override { return true; }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "LatentTexture[" << std::endl
            << "  texture = " << string::indent(m_texture) << "," << std::endl
            << "  tex_color  = " << string::indent(m_color)  << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(LatentTexture)
protected:
    ref<Texture> m_texture;
    ref<Texture> m_color;

    MI_TRAVERSE_CB(Texture, m_texture, m_color)
};

MI_EXPORT_PLUGIN(LatentTexture)
NAMESPACE_END(mitsuba)
