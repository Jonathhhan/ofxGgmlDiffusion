#include "ofxGgmlDiffusionUtils.h"

namespace ofxGgmlDiffusionUtils {
	bool hasInput(const ofxGgmlDiffusionRequest & request) {
		return !request.prompt.empty();
	}

	std::string describe(const ofxGgmlDiffusionRequest & request) {
		if (!hasInput(request)) {
			return "diffusion: empty request";
		}
		return "diffusion: " + request.prompt;
	}
}