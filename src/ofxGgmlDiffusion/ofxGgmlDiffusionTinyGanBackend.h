#pragma once

#include "ofxGgmlDiffusionImageGenerationBackend.h"

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

