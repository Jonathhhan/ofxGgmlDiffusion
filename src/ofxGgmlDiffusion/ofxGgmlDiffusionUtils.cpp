#include "ofxGgmlDiffusionUtils.h"

#include <cmath>
#include <sstream>

namespace ofxGgmlDiffusionUtils {
	bool hasInput(const ofxGgmlDiffusionRequest & request) {
		return !cleanPrompt(request.prompt).empty();
	}

	std::string describe(const ofxGgmlDiffusionRequest & request) {
		if (!hasInput(request)) {
			return "diffusion: empty request";
		}
		std::ostringstream stream;
		stream << "diffusion " << getModeName(request.mode) << " "
			<< request.width << "x" << request.height
			<< ": " << cleanPrompt(request.prompt);
		if (request.identityAdapter.isConfigured()) {
			stream << " [" << getIdentityAdapterTypeName(request.identityAdapter.type) << "]";
		}
		return stream.str();
	}

	std::string getModeName(ofxGgmlDiffusionMode mode) {
		switch (mode) {
		case ofxGgmlDiffusionMode::TextToImage: return "text-to-image";
		case ofxGgmlDiffusionMode::ImageToImage: return "image-to-image";
		case ofxGgmlDiffusionMode::Inpainting: return "inpainting";
		case ofxGgmlDiffusionMode::ImageToVideo: return "image-to-video";
		case ofxGgmlDiffusionMode::Upscale: return "upscale";
		default: return "unknown";
		}
	}

	std::string getModelFamilyName(ofxGgmlDiffusionModelFamily family) {
		switch (family) {
		case ofxGgmlDiffusionModelFamily::SD1: return "sd1";
		case ofxGgmlDiffusionModelFamily::SD2: return "sd2";
		case ofxGgmlDiffusionModelFamily::SDXL: return "sdxl";
		case ofxGgmlDiffusionModelFamily::SD3: return "sd3";
		case ofxGgmlDiffusionModelFamily::Flux: return "flux";
		case ofxGgmlDiffusionModelFamily::FluxFill: return "flux-fill";
		case ofxGgmlDiffusionModelFamily::FluxControl: return "flux-control";
		case ofxGgmlDiffusionModelFamily::Wan: return "wan";
		case ofxGgmlDiffusionModelFamily::QwenImage: return "qwen-image";
		default: return "unknown";
		}
	}

	std::string getIdentityAdapterTypeName(ofxGgmlDiffusionIdentityAdapterType type) {
		switch (type) {
		case ofxGgmlDiffusionIdentityAdapterType::PhotoMaker: return "photomaker";
		default: return "unknown";
		}
	}

	std::string cleanPrompt(const std::string & prompt) {
		std::string cleaned;
		cleaned.reserve(prompt.size());
		bool previousWasSpace = true;
		for (char value : prompt) {
			const bool isSpace = value == ' ' || value == '\t' || value == '\r' || value == '\n';
			if (isSpace) {
				if (!previousWasSpace) {
					cleaned.push_back(' ');
				}
				previousWasSpace = true;
			} else {
				cleaned.push_back(value);
				previousWasSpace = false;
			}
		}
		if (!cleaned.empty() && cleaned.back() == ' ') {
			cleaned.pop_back();
		}
		return cleaned;
	}

	bool isAutoValue(float value) {
		return std::isinf(value) && value > 0.0f;
	}

	bool isValidImageDimension(int value) {
		return value > 0 && value % 64 == 0;
	}

	ofxGgmlDiffusionValidationResult validate(const ofxGgmlDiffusionRequest & request) {
		ofxGgmlDiffusionValidationResult result;
		if (!hasInput(request)) {
			result.errors.push_back("prompt is empty");
		}
		if (!isValidImageDimension(request.width)) {
			result.errors.push_back("width must be a positive multiple of 64");
		}
		if (!isValidImageDimension(request.height)) {
			result.errors.push_back("height must be a positive multiple of 64");
		}
		if (request.batchCount < 1) {
			result.errors.push_back("batchCount must be at least 1");
		}
		if (request.batchCount > 16) {
			result.warnings.push_back("large batch counts can exhaust VRAM");
		}
		if (request.mode == ofxGgmlDiffusionMode::ImageToImage && request.initImagePath.empty()) {
			result.errors.push_back("image-to-image requires initImagePath");
		}
		if (request.mode == ofxGgmlDiffusionMode::Inpainting &&
			(request.initImagePath.empty() || request.maskImagePath.empty())) {
			result.errors.push_back("inpainting requires initImagePath and maskImagePath");
		}
		if (request.mode == ofxGgmlDiffusionMode::Upscale && request.initImagePath.empty()) {
			result.errors.push_back("upscale requires initImagePath");
		}
		if (!isAutoValue(request.cfgScale) && !std::isfinite(request.cfgScale)) {
			result.errors.push_back("cfgScale must be finite or auto");
		}
		if (!isAutoValue(request.strength) && !std::isfinite(request.strength)) {
			result.errors.push_back("strength must be finite or auto");
		}
		if (request.identityAdapter.isConfigured()) {
			const auto & adapter = request.identityAdapter;
			if (adapter.type == ofxGgmlDiffusionIdentityAdapterType::Unknown) {
				result.errors.push_back("identity adapter type is unknown");
			}
			if (adapter.modelPath.empty()) {
				result.errors.push_back("identity adapter requires modelPath");
			}
			if (!adapter.hasReferenceImages()) {
				result.errors.push_back("identity adapter requires at least one reference image");
			}
			if (adapter.triggerWord.empty()) {
				result.errors.push_back("identity adapter requires triggerWord");
			}
			if (!std::isfinite(adapter.strength) || adapter.strength < 0.0f) {
				result.errors.push_back("identity adapter strength must be finite and non-negative");
			}
			if (adapter.type == ofxGgmlDiffusionIdentityAdapterType::PhotoMaker &&
				request.mode != ofxGgmlDiffusionMode::TextToImage &&
				request.mode != ofxGgmlDiffusionMode::ImageToImage) {
				result.errors.push_back("PhotoMaker supports text-to-image or image-to-image requests");
			}
			if (adapter.type == ofxGgmlDiffusionIdentityAdapterType::PhotoMaker &&
				request.modelFamily != ofxGgmlDiffusionModelFamily::Unknown &&
				request.modelFamily != ofxGgmlDiffusionModelFamily::SDXL) {
				result.warnings.push_back("PhotoMaker is expected to run with SDXL-family diffusion models");
			}
		}
		result.success = result.errors.empty();
		return result;
	}

	ofxGgmlDiffusionRequest makeTextToImageRequest(const std::string & prompt) {
		ofxGgmlDiffusionRequest request;
		request.mode = ofxGgmlDiffusionMode::TextToImage;
		request.prompt = cleanPrompt(prompt);
		return request;
	}

	ofxGgmlDiffusionIdentityAdapter makePhotoMakerAdapter(
		const std::string & modelPath,
		const std::vector<std::string> & referenceImagePaths,
		const std::string & triggerWord) {
		ofxGgmlDiffusionIdentityAdapter adapter;
		adapter.type = ofxGgmlDiffusionIdentityAdapterType::PhotoMaker;
		adapter.modelPath = modelPath;
		adapter.referenceImagePaths = referenceImagePaths;
		adapter.triggerWord = cleanPrompt(triggerWord);
		return adapter;
	}
}
