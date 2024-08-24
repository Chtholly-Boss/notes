# STL Memory
```cpp
// Both std::unique_ptr and
// std::shared_ptr handle memory allocation and deallocation automatically, and
// contain raw pointers under the hood. In other words, they are wrapper classes
// over raw pointers
```
## unique_ptr
```cpp
// Includes std::unique_ptr functionality.
#include <memory>
// String library for printing help for demo purposes.
#include <string>
// Including the utility header for std::move.
#include <utility>

class Point {
public:
  Point() : x_(0), y_(0) {}
  Point(int x, int y) : x_(x), y_(y) {}
  inline int GetX() { return x_; }
  inline int GetY() { return y_; }
  inline void SetX(int x) { x_ = x; }
  inline void SetY(int y) { y_ = y; }

private:
  int x_;
  int y_;
};

void SetXTo445(std::unique_ptr<Point> &ptr) { ptr->SetX(445); }

// This is how to initialize an empty unique pointer of type
// std::unique_ptr<Point>.
std::unique_ptr<Point> u1;
// This is how to initialize a unique pointer with the default constructor.
std::unique_ptr<Point> u2 = std::make_unique<Point>();
// This is how to initialize a unique pointer with a custom constructor.
std::unique_ptr<Point> u3 = std::make_unique<Point>(2, 3);

// the statement (u ? "not empty" : "empty";
// to determine if the pointer u contains managed data. The main
// gist of this is that the std::unique_ptr class has a conversion function on
// its objects to a boolean type, and so this function is called whenever we
// treat the std::unique_ptr as a boolean.
if (u1) {
  // This won't print because u1 is empty.
  std::cout << "u1's value of x is " << u1->GetX() << std::endl;
}

if (u2) {
  // This will print because u2 is not empty, and contains a managed Point
  // instance.
  std::cout << "u2's value of x is " << u2->GetX() << std::endl;
}
// unique_ptr couldn't be copied
std::unique_ptr<Point> u4 = u3; // compilation error
// but can transfer ownership
std::unique_ptr<Point> u4 = std::move(u3);
```

## shared_ptr
```cpp
// std::shared_ptr is a type of smart pointer that retains shared ownership of
// an object through a pointer. This means that multiple shared pointers can
// own the same object, and shared pointers can be copied.
// Includes std::shared_ptr functionality.
#include <memory>
// Includes the utility header for std::move.
#include <utility>
class Point {
  // the same as in unique_ptr
  ...
}

void modify_ptr_via_ref(std::shared_ptr<Point> &point) { point->SetX(15); }

void modify_ptr_via_rvalue_ref(std::shared_ptr<Point> &&point) {
  point->SetY(645);
}

void copy_shared_ptr_in_function(std::shared_ptr<Point> point) {
  std::cout << "Use count of shared pointer is " << point.use_count()
            << std::endl;
}

// declaration and init like unique_ptr
std::shared_ptr<Point> s1;
std::shared_ptr<Point> s2 = std::make_shared<Point>();
std::shared_ptr<Point> s3 = std::make_shared<Point>(2, 3);
// check empty like unique_ptr
std::cout << "Pointer s1 is " << (s1 ? "not empty" : "empty") << std::endl;
std::cout << "Pointer s2 is " << (s2 ? "not empty" : "empty") << std::endl;
std::cout << "Pointer s3 is " << (s3 ? "not empty" : "empty") << std::endl;
// when copied, the ref_cnt increment
s3.use_count(); // which == 1 now 
// copy assignment
std::shared_ptr<Point> s4 = s3;
s3.use_count(); // which == 2 now 
// copy constructor
std::shared_ptr<Point> s5(s4);
s3.use_count(); // which == 3 now
// now s3,s4,s5 share the same data
// if one of them modify the data, all of them can see the change
s3->SetX(445);  // then s4.Get(X) == s5.Get(X) == 445
// shared ptr can be passed by ref/rvalue/value
// if passed by value, the callee copies the ptr
```
