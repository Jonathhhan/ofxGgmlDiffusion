#pragma once

#include "ofxGgmlDiffusionTypes.h"

#include "ofPixels.h"

#include <string>

namespace ofxGgmlDiffusionImageUtils {
	bool loadImage(const std::string& path, ofxGgmlDiffusionImage& image);
	bool toPixels(const ofxGgmlDiffusionImage& image, ofPixels& pixels);
	ofxGgmlDiffusionImage fromPixels(const ofPixels& pixels);
	bool saveImage(const ofxGgmlDiffusionImage& image, const std::string& path);
	bool saveFirstImage(ofxGgmlDiffusionResult& result, const std::string& path);
}
