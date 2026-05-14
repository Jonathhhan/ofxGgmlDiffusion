#pragma once

#include "ofxGgmlDiffusionTypes.h"

#include <string>

namespace ofxGgmlDiffusionUtils {
	bool hasInput(const ofxGgmlDiffusionRequest & request);
	std::string describe(const ofxGgmlDiffusionRequest & request);
	std::string getModeName(ofxGgmlDiffusionMode mode);
	std::string getModelFamilyName(ofxGgmlDiffusionModelFamily family);
	std::string getIdentityAdapterTypeName(ofxGgmlDiffusionIdentityAdapterType type);
	std::string getBackendFamilyName(ofxGgmlDiffusionBackendFamily family);
	std::string cleanPrompt(const std::string & prompt);
	bool isAutoValue(float value);
	bool isValidImageDimension(int value);
	ofxGgmlDiffusionValidationResult validate(const ofxGgmlDiffusionRequest & request);
	ofxGgmlDiffusionRequest makeTextToImageRequest(const std::string & prompt);
	ofxGgmlDiffusionRequest makeImageToVideoRequest(
		const std::string & prompt,
		const std::string & initImagePath,
		int videoFrameCount = 16);
	ofxGgmlDiffusionRequest makeGanImageRequest(
		const std::string & prompt,
		const std::string & generatorPath);
	ofxGgmlDiffusionIdentityAdapter makePhotoMakerAdapter(
		const std::string & modelPath,
		const std::vector<std::string> & referenceImagePaths,
		const std::string & triggerWord = "img");
}
