#pragma once

#include "ofxGgmlDiffusionImageGenerationBackend.h"

#include <cstdint>
#include <string>

struct ofxGgmlDiffusionTinyGanPreset {
	int version = 1;
	std::string architecture = "tiny-mlp";
	int latentSize = 512;
	int hiddenSize = 96;
	std::uint32_t w1Seed = 17;
	std::uint32_t b1Seed = 29;
	std::uint32_t w2Seed = 43;
	std::uint32_t b2Seed = 71;
	float latentScale = 1.0f;
	float w1Scale = 0.18f;
	float b1Scale = 0.08f;
	float w2Scale = 0.09f;
	float b2Scale = 0.03f;
};

ofxGgmlDiffusionTinyGanPreset ofxGgmlDiffusionMakeDefaultTinyGanPreset();
bool ofxGgmlDiffusionLoadTinyGanPreset(
	const std::string& path,
	ofxGgmlDiffusionTinyGanPreset& preset,
	std::string& error);
std::string ofxGgmlDiffusionSerializeTinyGanPreset(
	const ofxGgmlDiffusionTinyGanPreset& preset);

class ofxGgmlDiffusionTinyGanBackend : public ofxGgmlDiffusionImageGenerationBackend {
public:
	std::string getBackendName() const override;
	ofxGgmlDiffusionBackendFamily getBackendFamily() const override;
	bool isAvailable() const override;
	bool isLoaded() const override;

	ofxGgmlDiffusionResult setup(const ofxGgmlDiffusionContextSettings& settings) override;
	ofxGgmlDiffusionResult generate(const ofxGgmlDiffusionRequest& request) override;

private:
	bool loaded = false;
	int threads = 1;
};
