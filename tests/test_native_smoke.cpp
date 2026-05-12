#include "ofxGgmlDiffusion/ofxGgmlDiffusionNativeBackend.h"
#include "ofxGgmlDiffusion/ofxGgmlDiffusionUtils.h"

#include <iostream>
#include <string>

int main() {
	ofxGgmlDiffusionNativeBackend backend;
	if (backend.getBackendName() != "stable-diffusion.cpp") {
		std::cerr << "unexpected native backend name\n";
		return 1;
	}
	if (!backend.isAvailable()) {
		std::cerr << "stable-diffusion.cpp backend is not available in this build\n";
		return 1;
	}
	if (backend.isLoaded()) {
		std::cerr << "native backend reported loaded before setup\n";
		return 1;
	}

	const auto setupResult = backend.setup(ofxGgmlDiffusionContextSettings{});
	if (setupResult.isOk() ||
		setupResult.error.find("no diffusion model path") == std::string::npos) {
		std::cerr << "empty native setup returned unexpected result: " << setupResult.error << "\n";
		return 1;
	}

	auto request = ofxGgmlDiffusionUtils::makeTextToImageRequest("small local test image");
	request.width = 64;
	request.height = 64;
	request.steps = 1;
	const auto generateResult = backend.generate(request);
	if (generateResult.isOk() ||
		generateResult.error.find("context is not loaded") == std::string::npos) {
		std::cerr << "native generation without setup returned unexpected result: " << generateResult.error << "\n";
		return 1;
	}

	return 0;
}
