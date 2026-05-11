#include "ofxGgmlDiffusionImageGenerationBackend.h"

namespace {
	ofxGgmlDiffusionResult makeUnavailableResult(const std::string& name) {
		ofxGgmlDiffusionResult result;
		result.success = false;
		result.error = name + " image generation backend is not available";
		return result;
	}
}

ofxGgmlDiffusionUnavailableImageGenerationBackend::ofxGgmlDiffusionUnavailableImageGenerationBackend(
	ofxGgmlDiffusionBackendFamily family,
	const std::string& name)
	: family(family)
	, name(name.empty() ? "unavailable" : name) {
}

std::string ofxGgmlDiffusionUnavailableImageGenerationBackend::getBackendName() const {
	return name;
}

ofxGgmlDiffusionBackendFamily ofxGgmlDiffusionUnavailableImageGenerationBackend::getBackendFamily() const {
	return family;
}

bool ofxGgmlDiffusionUnavailableImageGenerationBackend::isAvailable() const {
	return false;
}

bool ofxGgmlDiffusionUnavailableImageGenerationBackend::isLoaded() const {
	return false;
}

ofxGgmlDiffusionResult ofxGgmlDiffusionUnavailableImageGenerationBackend::setup(
	const ofxGgmlDiffusionContextSettings& settings) {
	auto result = makeUnavailableResult(name);
	if (!settings.modelPath.empty()) {
		result.references.push_back(settings.modelPath);
	}
	if (!settings.diffusionModelPath.empty()) {
		result.references.push_back(settings.diffusionModelPath);
	}
	return result;
}

ofxGgmlDiffusionResult ofxGgmlDiffusionUnavailableImageGenerationBackend::generate(
	const ofxGgmlDiffusionRequest& request) {
	auto result = makeUnavailableResult(name);
	result.outputPath = request.outputPath;
	result.seed = request.seed;
	if (!request.gan.generatorPath.empty()) {
		result.references.push_back(request.gan.generatorPath);
	}
	return result;
}

std::unique_ptr<ofxGgmlDiffusionImageGenerationBackend>
ofxGgmlMakeUnavailableDiffusionImageGenerationBackend(
	ofxGgmlDiffusionBackendFamily family,
	const std::string& name) {
	return std::make_unique<ofxGgmlDiffusionUnavailableImageGenerationBackend>(family, name);
}
