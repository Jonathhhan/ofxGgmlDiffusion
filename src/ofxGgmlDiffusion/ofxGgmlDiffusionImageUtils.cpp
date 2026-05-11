#include "ofxGgmlDiffusionImageUtils.h"

#include "ofImage.h"
#include "ofLog.h"

namespace ofxGgmlDiffusionImageUtils {
	namespace {
		ofImageType getImageType(int channels) {
			switch (channels) {
			case 1: return OF_IMAGE_GRAYSCALE;
			case 3: return OF_IMAGE_COLOR;
			case 4: return OF_IMAGE_COLOR_ALPHA;
			default: return OF_IMAGE_UNDEFINED;
			}
		}
	}

	bool toPixels(const ofxGgmlDiffusionImage& image, ofPixels& pixels) {
		pixels.clear();
		if (!image.isAllocated()) {
			return false;
		}
		const auto imageType = getImageType(image.channels);
		if (imageType == OF_IMAGE_UNDEFINED) {
			ofLogWarning("ofxGgmlDiffusion") << "unsupported image channel count: " << image.channels;
			return false;
		}
		const auto expectedBytes =
			static_cast<std::size_t>(image.width) *
			static_cast<std::size_t>(image.height) *
			static_cast<std::size_t>(image.channels);
		if (image.pixels.size() != expectedBytes) {
			ofLogWarning("ofxGgmlDiffusion") << "image byte size does not match dimensions";
			return false;
		}
		pixels.setFromPixels(
			image.pixels.data(),
			image.width,
			image.height,
			imageType);
		return pixels.isAllocated();
	}

	ofxGgmlDiffusionImage fromPixels(const ofPixels& pixels) {
		ofxGgmlDiffusionImage image;
		if (!pixels.isAllocated()) {
			return image;
		}
		image.width = pixels.getWidth();
		image.height = pixels.getHeight();
		image.channels = pixels.getNumChannels();
		image.pixels.assign(pixels.getData(), pixels.getData() + pixels.size());
		return image;
	}

	bool saveImage(const ofxGgmlDiffusionImage& image, const std::string& path) {
		ofPixels pixels;
		if (!toPixels(image, pixels)) {
			return false;
		}
		return ofSaveImage(pixels, path);
	}

	bool saveFirstImage(ofxGgmlDiffusionResult& result, const std::string& path) {
		if (result.images.empty()) {
			result.error = "no generated image was available to save";
			return false;
		}
		if (!saveImage(result.images.front(), path)) {
			result.error = "failed to save generated image: " + path;
			return false;
		}
		result.outputPath = path;
		result.imagePaths.clear();
		result.imagePaths.push_back(path);
		return true;
	}
}
