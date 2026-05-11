#pragma once

#include "ofxGgmlDiffusionTypes.h"

#include <string>

namespace ofxGgmlDiffusionUtils {
	bool hasInput(const ofxGgmlDiffusionRequest & request);
	std::string describe(const ofxGgmlDiffusionRequest & request);
}