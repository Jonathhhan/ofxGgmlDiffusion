#include "ofxGgmlDiffusion.h"

#include <iostream>

int main() {
	ofxGgmlDiffusionRequest request;
	if (ofxGgmlDiffusionUtils::hasInput(request)) {
		std::cerr << "empty request reported as configured\n";
		return 1;
	}

	request.prompt = "a small generative texture study";
	if (!ofxGgmlDiffusionUtils::hasInput(request)) {
		std::cerr << "configured request reported as empty\n";
		return 1;
	}

	const auto description = ofxGgmlDiffusionUtils::describe(request);
	if (description.find(request.prompt) == std::string::npos) {
		std::cerr << "description did not include request input\n";
		return 1;
	}

	return 0;
}