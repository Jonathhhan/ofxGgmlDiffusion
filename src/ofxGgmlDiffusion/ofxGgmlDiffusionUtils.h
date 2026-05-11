#pragma once

#include "ofxGgmlDiffusionTypes.h"

#include <string>

namespace ofxGgmlDiffusionUtils {
	bool hasInput(const ofxGgmlDiffusionRequest & request);
	std::string describe(const ofxGgmlDiffusionRequest & request);
	std::string getModeName(ofxGgmlDiffusionMode mode);
	std::string getModelFamilyName(ofxGgmlDiffusionModelFamily family);
	std::string cleanPrompt(const std::string & prompt);
	bool isAutoValue(float value);
	bool isValidImageDimension(int value);
	ofxGgmlDiffusionValidationResult validate(const ofxGgmlDiffusionRequest & request);
	ofxGgmlDiffusionRequest makeTextToImageRequest(const std::string & prompt);
}
