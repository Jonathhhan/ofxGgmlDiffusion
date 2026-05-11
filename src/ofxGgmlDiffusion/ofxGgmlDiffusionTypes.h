#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

enum class ofxGgmlDiffusionMode {
	TextToImage = 0,
	ImageToImage,
	Inpainting,
	ImageToVideo,
	Upscale
};

enum class ofxGgmlDiffusionModelFamily {
	Unknown = 0,
	SD1,
	SD2,
	SDXL,
	SD3,
	Flux,
	FluxFill,
	FluxControl,
	Wan,
	QwenImage
};

enum class ofxGgmlDiffusionScheduler {
	Auto = 0,
	Default,
	LCM,
	Turbo,
	FlowMatch
};

struct ofxGgmlDiffusionLora {
	std::string path;
	float strength = 1.0f;
	bool highNoise = false;

	bool isConfigured() const {
		return !path.empty();
	}
};

struct ofxGgmlDiffusionControlImage {
	std::string imagePath;
	std::string type;
	float strength = 0.9f;

	bool isConfigured() const {
		return !imagePath.empty();
	}
};

struct ofxGgmlDiffusionContextSettings {
	std::string modelPath;
	std::string diffusionModelPath;
	std::string clipLPath;
	std::string clipGPath;
	std::string t5xxlPath;
	std::string vaePath;
	std::string taesdPath;
	std::string loraDirectory;
	std::string embeddingDirectory;
	std::string upscalerModelPath;
	ofxGgmlDiffusionModelFamily modelFamily = ofxGgmlDiffusionModelFamily::Unknown;
	int threads = -1;
	bool vaeTiling = false;
	bool flashAttention = false;
	bool mmap = true;

	bool hasAnyModelPath() const {
		return !modelPath.empty() ||
			!diffusionModelPath.empty() ||
			!upscalerModelPath.empty();
	}
};

struct ofxGgmlDiffusionRequest {
	ofxGgmlDiffusionMode mode = ofxGgmlDiffusionMode::TextToImage;
	std::string prompt;
	std::string negativePrompt;
	std::string initImagePath;
	std::string maskImagePath;
	std::string outputPath;
	int width = 512;
	int height = 512;
	int steps = -1;
	float cfgScale = std::numeric_limits<float>::infinity();
	float strength = std::numeric_limits<float>::infinity();
	float flowShift = std::numeric_limits<float>::infinity();
	std::int64_t seed = -1;
	int batchCount = 1;
	ofxGgmlDiffusionScheduler scheduler = ofxGgmlDiffusionScheduler::Auto;
	std::vector<ofxGgmlDiffusionLora> loras;
	std::vector<ofxGgmlDiffusionControlImage> controlImages;
	std::vector<std::string> tags;
};

struct ofxGgmlDiffusionResult {
	bool success = false;
	std::string text;
	std::string error;
	std::string outputPath;
	std::vector<std::string> imagePaths;
	std::vector<std::string> references;
	float elapsedMs = 0.0f;
	std::int64_t seed = -1;

	explicit operator bool() const {
		return success;
	}
};

struct ofxGgmlDiffusionValidationResult {
	bool success = true;
	std::vector<std::string> errors;
	std::vector<std::string> warnings;

	explicit operator bool() const {
		return success;
	}
};
