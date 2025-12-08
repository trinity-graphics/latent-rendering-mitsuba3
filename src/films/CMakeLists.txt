set(MI_PLUGIN_PREFIX "films")

add_plugin(hdrfilm  hdrfilm.cpp)
add_plugin(specfilm  specfilm.cpp)
add_plugin(latfilm  latfilm.cpp)

set(MI_PLUGIN_TARGETS "${MI_PLUGIN_TARGETS}" PARENT_SCOPE)
