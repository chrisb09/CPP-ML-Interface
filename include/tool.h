#ifndef TOOL_H
#define TOOL_H

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
    
// Pure C assertion helper
void guarantee(bool condition, const char* message) {
    if (!condition) {
        fprintf(stderr, "Assertion failed: %s\n", message);
        exit(EXIT_FAILURE);
    }
}


#endif // TOOL_H