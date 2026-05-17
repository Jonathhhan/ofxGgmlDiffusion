#pragma once

#include "ofxGgmlDiffusionImageGenerationBackend.h"

#if __has_include("ofxGgmlCore.h")
#include "ofxGgmlCore.h"
#elif __has_include("../../../ofxGgmlCore/src/ofxGgmlCore.h")
#include "../../../ofxGgmlCore/src/ofxGgmlCore.h"
#elif __has_include("../../../../ofxGgmlCore/src/ofxGgmlCore.h")
#include "../../../../ofxGgmlCore/src/ofxGgmlCore.h"
#else
#error "Cannot find ofxGgmlCore.h. Ensure the ofxGgmlCore addon is linked and available as a sibling addon."
#endif

#include <cstdint>
#include <string>

class ofxGgmlDiffusionGgufGanBackend : public ofxGgmlDiffusionImageGenerationBackend {
public:
	std::string getBackendName() const override;
	ofxGgmlDiffusionBackendFamily getBackendFamily() const override;
	bool isAvailable() const override;
	bool isLoaded() const override;

	ofxGgmlDiffusionResult setup(const ofxGgmlDiffusionContextSettings& settings) override;
	ofxGgmlDiffusionResult generate(const ofxGgmlDiffusionRequest& request) override;

private:
	bool loaded = false;
	std::string loadedGeneratorPath;
	ofxGgmlModelInfo modelInfo{};
};
