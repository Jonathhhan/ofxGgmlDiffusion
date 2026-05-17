meta:
	ADDON_NAME = ofxGgmlDiffusion
	ADDON_DESCRIPTION = Companion addon for local diffusion image/video generation workflows on top of ofxGgmlCore
	ADDON_AUTHOR = Jonathan Frank
	ADDON_TAGS = "ggml,ai,diffusion,image-generation,creative-coding"
	ADDON_URL = https://github.com/Jonathhhan/ofxGgmlDiffusion

common:
	ADDON_DEPENDENCIES += ofxGgmlCore
	ADDON_INCLUDES += src
	ADDON_INCLUDES += libs/stable-diffusion/include
	# Native stable-diffusion.cpp bridge is auto-managed by scripts/build-stable-diffusion.*
	# ADDON_CFLAGS += -DOFXGGMLDIFFUSION_WITH_STABLE_DIFFUSION
	ADDON_SOURCES_EXCLUDE += build/%
	ADDON_SOURCES_EXCLUDE += libs/stable-diffusion/.source/%
	ADDON_SOURCES_EXCLUDE += libs/stable-diffusion/build/%
	ADDON_SOURCES_EXCLUDE += libs/stable-diffusion/build*/%
	ADDON_SOURCES_EXCLUDE += libs/*/build/%
	ADDON_SOURCES_EXCLUDE += libs/*/build*/%
	ADDON_INCLUDES_EXCLUDE += build/%
	ADDON_INCLUDES_EXCLUDE += libs/stable-diffusion/.source/%
	ADDON_INCLUDES_EXCLUDE += libs/stable-diffusion/build/%
	ADDON_INCLUDES_EXCLUDE += libs/stable-diffusion/build*/%
	ADDON_INCLUDES_EXCLUDE += libs/*/build/%
	ADDON_INCLUDES_EXCLUDE += libs/*/build*/%

vs:
	# Local stable-diffusion.cpp runtime linkage (auto-managed by build-stable-diffusion.*):
	# ADDON_LIBS += libs/stable-diffusion/lib/stable-diffusion.lib

linux64:
	# Local stable-diffusion.cpp runtime linkage (auto-managed by build-stable-diffusion.*):
	# ADDON_LIBS += libs/stable-diffusion/lib/libstable-diffusion.a

osx:
	# Local stable-diffusion.cpp runtime linkage (auto-managed by build-stable-diffusion.*):
	# ADDON_LIBS += libs/stable-diffusion/lib/libstable-diffusion.a
