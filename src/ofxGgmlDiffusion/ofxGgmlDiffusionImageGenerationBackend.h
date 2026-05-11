#pragma once

#include "ofxGgmlDiffusionTypes.h"

#include <memory>
#include <string>

class ofxGgmlDiffusionImageGenerationBackend {
public:
	virtual ~ofxGgmlDiffusionImageGenerationBackend() = default;

	virtual std::string getBackendName() const = 0;
	virtual ofxGgmlDiffusionBackendFamily getBackendFamily() const = 0;
	virtual bool isAvailable() const = 0;
	virtual bool isLoaded() const = 0;

	virtual ofxGgmlDiffusionResult setup(const ofxGgmlDiffusionContextSettings& settings) = 0;
	virtual ofxGgmlDiffusionResult generate(const ofxGgmlDiffusionRequest& request) = 0;
};

class ofxGgmlDiffusionUnavailableImageGenerationBackend
	: public ofxGgmlDiffusionImageGenerationBackend {
public:
	ofxGgmlDiffusionUnavailableImageGenerationBackend(
		ofxGgmlDiffusionBackendFamily family = ofxGgmlDiffusionBackendFamily::Auto,
		const std::string& name = "unavailable");

	std::string getBackendName() const override;
	ofxGgmlDiffusionBackendFamily getBackendFamily() const override;
	bool isAvailable() const override;
	bool isLoaded() const override;

	ofxGgmlDiffusionResult setup(const ofxGgmlDiffusionContextSettings& settings) override;
	ofxGgmlDiffusionResult generate(const ofxGgmlDiffusionRequest& request) override;

private:
	ofxGgmlDiffusionBackendFamily family = ofxGgmlDiffusionBackendFamily::Auto;
	std::string name;
};

std::unique_ptr<ofxGgmlDiffusionImageGenerationBackend>
ofxGgmlMakeUnavailableDiffusionImageGenerationBackend(
	ofxGgmlDiffusionBackendFamily family = ofxGgmlDiffusionBackendFamily::Auto,
	const std::string& name = "unavailable");
