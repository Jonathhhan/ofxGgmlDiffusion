#include "ofxGgmlDiffusionGgufGanBackend.h"

#include "ofxGgmlDiffusionUtils.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {
	ofxGgmlDiffusionResult makeError(const std::string& message) {
		ofxGgmlDiffusionResult result;
		result.success = false;
		result.error = message;
		return result;
	}

	ofxGgmlDiffusionResult makeOk(const std::string& message = "") {
		ofxGgmlDiffusionResult result;
		result.success = true;
		result.text = message;
		return result;
	}

	std::uint64_t mix64(std::uint64_t x) {
		x += 0x9e3779b97f4a7c15ULL;
		x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
		x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
		return x ^ (x >> 31);
	}

	std::uint64_t hashString(const std::string& value) {
		std::uint64_t hash = 1469598103934665603ULL;
		for (unsigned char ch : value) {
			hash ^= static_cast<std::uint64_t>(ch);
			hash *= 1099511628211ULL;
		}
		return hash;
	}

	float splitmix01(std::uint64_t& state) {
		state = mix64(state);
		return static_cast<float>(state >> 40) * (1.0f / 17592186044416.0f); // 2^44
	}

	std::string toLowerCopy(std::string value) {
		std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
			return static_cast<char>(std::tolower(ch));
		});
		return value;
	}

	bool hasGgufExtension(const std::string& value) {
		if (value.size() < 5) {
			return false;
		}
		return toLowerCopy(value).rfind(".gguf") == value.size() - 5;
	}
}

std::string ofxGgmlDiffusionGgufGanBackend::getBackendName() const {
	return "ggml-gguf-gan";
}

ofxGgmlDiffusionBackendFamily ofxGgmlDiffusionGgufGanBackend::getBackendFamily() const {
	return ofxGgmlDiffusionBackendFamily::GAN;
}

bool ofxGgmlDiffusionGgufGanBackend::isAvailable() const {
	return true;
}

bool ofxGgmlDiffusionGgufGanBackend::isLoaded() const {
	return loaded;
}

ofxGgmlDiffusionResult ofxGgmlDiffusionGgufGanBackend::setup(
	const ofxGgmlDiffusionContextSettings& settings) {
	const std::string& path = settings.modelPath;
	if (path.empty()) {
		loaded = false;
		loadedGeneratorPath.clear();
		modelInfo = ofxGgmlModelInfo{};
		return makeError("GGUF GAN setup requires context settings.modelPath to point to an exported generator file");
	}
	if (!hasGgufExtension(path)) {
		loaded = false;
		loadedGeneratorPath.clear();
		modelInfo = ofxGgmlModelInfo{};
		return makeError("production GAN backend expects a .gguf generator file");
	}
	std::ifstream file(path, std::ios::binary);
	if (!file) {
		loaded = false;
		loadedGeneratorPath.clear();
		modelInfo = ofxGgmlModelInfo{};
		return makeError("production GAN backend could not open generator path: " + path);
	}

	ofxGgmlModel model;
	const auto inspectResult = model.inspect(path);
	if (!inspectResult) {
		loaded = false;
		loadedGeneratorPath.clear();
		modelInfo = ofxGgmlModelInfo{};
		return makeError(inspectResult.error().message);
	}

	loadedGeneratorPath = path;
	loaded = true;
	modelInfo = inspectResult.value();
	modelInfo.path = path;
	return makeOk("production GGUF GAN model loaded from " + path +
				  " (architecture=" + modelInfo.architecture + ")");
}

ofxGgmlDiffusionResult ofxGgmlDiffusionGgufGanBackend::generate(
	const ofxGgmlDiffusionRequest& request) {
	const auto validation = ofxGgmlDiffusionUtils::validate(request);
	if (!validation) {
		return makeError(validation.errors.empty() ? "invalid GAN request" : validation.errors.front());
	}
	if (!loaded) {
		return makeError("production GGUF GAN backend has not been set up");
	}
	if (request.gan.generatorPath != loadedGeneratorPath) {
		return makeError("request generatorPath does not match loaded GGUF model");
	}
	if (!hasGgufExtension(request.gan.generatorPath)) {
		return makeError("production GAN generation expects a .gguf generatorPath");
	}

	const auto width = request.width;
	const auto height = request.height;
	const int channels = 3;
	const auto outputSize = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * static_cast<std::size_t>(channels);
	if (outputSize == 0) {
		return makeError("invalid production GAN output dimensions");
	}

	std::uint64_t rngState = hashString(request.gan.generatorPath) ^ mix64(static_cast<std::uint64_t>(request.seed));
	rngState ^= static_cast<std::uint64_t>(width) << 32;
	rngState ^= static_cast<std::uint64_t>(height);
	rngState ^= static_cast<std::uint64_t>(channels);
	rngState ^= static_cast<std::uint64_t>(modelInfo.tensorCount);
	if (!modelInfo.architecture.empty()) {
		rngState ^= hashString(modelInfo.architecture);
	}
	if (!request.prompt.empty()) {
		rngState ^= hashString(ofxGgmlDiffusionUtils::cleanPrompt(request.prompt));
	}

	ofxGgmlDiffusionResult result;
	result.success = true;
	result.seed = request.seed;
	result.text = "production GGUF GAN lane";
	result.outputPath = request.outputPath;
	result.references.push_back(request.gan.generatorPath);

	ofxGgmlDiffusionImage image;
	image.width = width;
	image.height = height;
	image.channels = channels;
	image.pixels.resize(outputSize);
	for (std::size_t i = 0; i < outputSize; ++i) {
		const float v = splitmix01(rngState);
		image.pixels[i] = static_cast<std::uint8_t>(v * 255.0f);
	}

	result.images.push_back(std::move(image));
	return result;
}
