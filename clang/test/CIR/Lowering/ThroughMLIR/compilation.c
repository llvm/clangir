// RUN: %clang -fclangir -fno-clangir-direct-lowering %s -o %t

int main(int argc, char *argv[]) {
    // Variables
    int number = 0;

    return number;
} 