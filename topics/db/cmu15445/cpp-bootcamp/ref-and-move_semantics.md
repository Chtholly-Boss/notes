# Reference and Move semantics
## Reference
```cpp
int a = 10;
// you can declare a refence like this
int &b = a;

// functions can receive reference
int c = 10
void add3(int &a){ a = a + 3;}
add3(c);    // c will become 13
/*
Reference Binding: When you declare a parameter as int&, 
you are telling the compiler that this parameter is a reference to an int. 
When you pass an int variable to this function, 
the reference binds to that variable, 
allowing the function to operate on the original variable rather than a copy.
*/
```
## move_semantics
```cpp
// Includes the utility header for std::move.
#include <utility>
#include <vector>
// Rvalue references are references that refer to the data itself, as opposed
// to a lvalue. Calling std::move on a lvalue (such as stealing_ints) will
// result in the expression being cast to a rvalue reference.
std::vector<int> &&rvalue_stealing_ints = std::move(stealing_ints);

// Once the rvalue is moved from the lvalue in the caller context to a lvalue
// in the callee context, it is effectively unusable to the caller.
void move_add_three_and_print(std::vector<int> &&vec) {
  std::vector<int> vec1 = std::move(vec);
  vec1.push_back(3);
}
// Essentially, after move_add_three_and_print is called, we cannot use the
// data in int_array2. It no longer belongs to the int_array2 lvalue.
std::vector<int> int_array2 = {1, 2, 3, 4};
move_add_three_and_print(std::move(int_array2));

// If we don't move the lvalue in the caller context to any lvalue in the
// callee context, then effectively the function treats the rvalue reference
// passed in as a reference, and the lvalue in this context still owns the
// vector data.
void add_three_and_print(std::vector<int> &&vec) {
  vec.push_back(3);
}
std::vector<int> int_array3 = {1, 2, 3, 4};
add_three_and_print(std::move(int_array3));

// && is a single operator that signifies an rvalue reference. 
// It is distinct from &, which denotes a regular lvalue reference.
```
## move_constructors
```cpp
// declare a class
class Person {
// access specifier declaration
public:
  // Constructor: the same name as the class name
  // content between : and {} is a member initializer list
  // it will be executed before execute the body
  Person() : age_(0), nicknames_({}), valid_(true) {}
  Person(uint32_t age, std::vector<std::string> &&nicknames)
      : age_(age), nicknames_(std::move(nicknames)), valid_(true) {}

  // move constructor
  // for numeric types, copy is ok
  // for string or object types, use move to achieve perf
  Person(Person &&person)
      : age_(person.age_), nicknames_(std::move(person.nicknames_)),
        valid_(true) {
    person.valid_ = false;
  }
  // operator overloading: must be one of the predefined operators in cpp
  // Move assignment operator
  Person &operator=(Person &&other) {
    age_ = other.age_;
    nicknames_ = std::move(other.nicknames_);
    valid_ = true;

    other.valid_ = false;
    // this is a pointer to the current object
    return *this;
  }

  // delete copy constructor
  // A copy constructor is a special constructor 
  // that initializes a new object as a copy of an existing object. 
  // The default copy constructor performs a 
  // member-wise copy of the object's data members.
  Person(const Person &) = delete;
  Person &operator=(const Person &) = delete;

  std::string &GetNicknameAtI(size_t i) { return nicknames_[i]; }

private:
// a underscore _ is a naming convention.
// it helps distinguish member variables from other variables
  uint32_t age_;
  std::vector<std::string> nicknames_;
  bool valid_;
};
```