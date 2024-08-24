# C++ Templates
## templated functions
```cpp
// declaration
template <typename T> T add(T a, T b) { return a + b; }

template<typename T, typename U>
void print_two_values(T a, U b) {
  std::cout << a << " and " << b << std::endl;
}

// specialized templates
// Prints the type if its a float type, but just prints hello world for
// all other types.
template <typename T> void print_msg() { std::cout << "Hello world!\n"; }
// Specialized templated function, specialized on the float type.
template <> void print_msg<float>() {
  std::cout << "print_msg called with float type!\n";
}
// template parameters do not have to be classes
// you can instantiate functions with different conditions 
// which might be useful to form different tests
template <bool T> int add3(int a) {
  if (T) {
    return a + 3;
  }

  return a;
}

// to instantiate a templated function
add<int>(3,5);
add3<true>(3);
```
## templated classes
```cpp
// declaration
// the same as templated functions
template<typename T>
class Foo {
  public:
    Foo(T var) : var_(var) {}
    void print() {
      std::cout << var_ << std::endl;
    }
  private:
    T var_;
};
```
