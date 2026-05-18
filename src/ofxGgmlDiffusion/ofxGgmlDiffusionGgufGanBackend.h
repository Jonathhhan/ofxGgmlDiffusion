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
#include <vector>

enum class ofxGgmlDiffusionGgufGanRuntimeKind {
	Unknown = 0,
	PixelDcgan
};

struct ofxGgmlDiffusionGgufGanPixelDcganModel {
	std::vector<float> fcWeight;
	std::vector<float> bn1Bias;
	std::vector<float> bn1Mean;
	std::vector<float> bn1Variance;
	std::vector<float> bn1Weight;
	std::vector<float> bn2Bias;
	std::vector<float> bn2Mean;
	std::vector<float> bn2Variance;
	std::vector<float> bn2Weight;
	std::vector<float> bn3Bias;
	std::vector<float> bn3Mean;
	std::vector<float> bn3Variance;
	std::vector<float> bn3Weight;
	std::vector<float> convTranspose1Weight;
	std::vector<float> convTranspose2Weight;
	std::vector<float> convTranspose3Weight;
};

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
	ofxGgmlDiffusionGgufGanRuntimeKind runtimeKind =
		ofxGgmlDiffusionGgufGanRuntimeKind::Unknown;
	ofxGgmlDiffusionGgufGanPixelDcganModel pixelDcganModel;
};
