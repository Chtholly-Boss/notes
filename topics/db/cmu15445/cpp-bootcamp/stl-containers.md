# STL Containers
## vector
```cpp
// Includes std::remove_if to remove elements from vectors.
#include <algorithm>
// Includes the vector container library header.
#include <vector>
class Point {
public:
  Point() : x_(0), y_(0) {
  }

  Point(int x, int y) : x_(x), y_(y) {
  }

  // When the compiler encounters an inline function call,
  // it may replace the call with the actual function code.
  // This can reduce the overhead associated with function calls
  inline int GetX() const { return x_; }
  inline int GetY() const { return y_; }
  inline void SetX(int x) { x_ = x; }
  inline void SetY(int y) { y_ = y; }
  void PrintPoint() const {
    std::cout << "Point value is (" << x_ << ", " << y_ << ")\n";
  }

private:
  int x_;
  int y_;
};

// We can declare a Point vector with the following syntax.
std::vector<Point> point_vector;

// It is also possible to initialize the vector via an initializer list.
std::vector<int> int_vector = {0, 1, 2, 3, 4, 5, 6};

// to append data
// 1. emplace_back forwards the constructor arguments
// to the object's constructor and 
// constructs the object in place
// 2. push_back constructs the object, then
// moves it to the memory in the vector.
std::cout << "Appending to the point_vector via push_back:\n";
point_vector.push_back(Point(35, 36));
std::cout << "Appending to the point_vector via emplace_back:\n";
point_vector.emplace_back(37, 38);
// to iterate on a vector
for (size_t i = 0; i < point_vector.size(); ++i) {
  point_vector[i].PrintPoint();
}
// for-each
for (Point &item : point_vector) {
  item.SetY(445);
}
// Note that I use the const reference
// syntax to ensure that the data I'm accessing is read only.
for (const Point &item : point_vector) {
  item.PrintPoint();
}

// to erase a element
int_vector.erase(int_vector.begin() + 2); // erase int_vector[2]
// to erase a range
int_vector.erase(int_vector.begin() + 1, int_vector.end()); // erase int_vector[1:]
// combine it with remove_if from algorithm
// remove_if takes a range and a lambda
// lambda parameter is the element in the range to be filted
// this will partition the vector into { ... | elem,to,be,deleted}
// and remove_if return the iterator points to position `elem`
point_vector.erase(
    std::remove_if(point_vector.begin(), point_vector.end(), 
                    [](const Point &point) { return point.GetX() == 37; }),
    point_vector.end());
```
## set
```cpp
// Includes the set container library header.
#include <set>
// declaration
std::set<int> int_set;
// insertion
for (int i = 1; i <= 5; ++i) {
  int_set.insert(i);
}
// emplace like vector
for (int i = 6; i <= 10; ++i) {
  int_set.emplace(i);
}
// find an element
std::set<int>::iterator search = int_set.find(2); // 2 is the key
// or use count
if (int_set.count(11) == 0) {
  std::cout << "Element 11 is not in the set.\n";
}

if (int_set.count(3) == 1) {
  std::cout << "Element 3 is in the set.\n";
}
// erase is similar as vector::erase
int_set.erase(4); // erase by key
int_set.erase(int_set.begin()); // erase by position
int_set.erase(int_set.find(9), int_set.end());  // erase by range
// since set is ordered, it erases elements >= 9

// iteration
// for-each
for (const int &elem : int_set) {
  std::cout << elem << " ";
}
// using iterator
for (std::set<int>::iterator it = int_set.begin(); it != int_set.end();
      ++it) {
  // We can access the element itself by dereferencing the iterator.
  std::cout << *it << " ";
}

```
## unordered map
```cpp
// Includes the unordered_map container library header.
#include <unordered_map>
// Includes the C++ string library.
#include <string>
// Includes std::make_pair.
#include <utility>

int main(){
  // declaration
  std::unordered_map<std::string, int> map;
  // insert a key-value pair
  map.insert({"foo", 2});
    // The insert function also takes in a std::pair as the argument. An
  // std::pair is a generic pair type, and you can create one by calling
  // std::make_pair with 2 arguments. std::make_pair is defined in the header
  // <utility>, and constructs an instance of the generic pair type.
  map.insert(std::make_pair("jignesh", 445));

  // You can also insert multiple elements at a time by passing in an
  // initializer list of pairs.
  map.insert({{"spam", 1}, {"eggs", 2}, {"garlic rice", 3}});
  // array-like syntax
  // if exists, update value
  // if not exists, insert 
  map["bacon"] = 5;
  // find like set
  std::unordered_map<std::string, int>::iterator result = map.find("jignesh");
  // when dereference it, get a pair
  std::pair<std::string, int> pair = *result;
  std::cout << "DEREF: Found key " << pair.first << " with value "
            << pair.second << std::endl;
  // count like set
  // erase like set
  map.erase(map.find("garlic rice"));
  if (map.count("garlic rice") == 0) {
    std::cout << "Key-value pair with key garlic rice does not exist in the "
                 "unordered map.\n";
  }
  // iteration also
  // using iterator
  for (std::unordered_map<std::string, int>::iterator it = map.begin();
       it != map.end(); ++it) {
    // We can access the element itself by dereferencing the iterator.
    std::cout << "(" << it->first << ", " << it->second << "), ";
  }
  std::cout << "\n";
  // for-each method
  for (const std::pair<const std::string, int> &elem : map) {
    std::cout << "(" << elem.first << ", " << elem.second << "), ";
  }
}
```

## auto
```cpp
// Includes the C++ string library.
#include <string>
// Includes the std::vector library.
#include <vector>
// Includes the std::unordered map library.
#include <unordered_map>
// The C++ auto keyword is a keyword that tells the compiler to infer the type
// of a declared variable via its initialization expression.

// The auto keyword is used to initialize the variable a. Here, the type
// is inferred to be type int.
auto a = 1;

// very long name to indicate the use of auto
template <typename T, typename U> class Abcdefghijklmnopqrstuvwxyz {
public:
  Abcdefghijklmnopqrstuvwxyz(T instance1, U instance2)
      : instance1_(instance1), instance2_(instance2) {}

  void print() const {
    std::cout << "(" << instance1_ << "," << instance2_ << ")\n";
  }

private:
  T instance1_;
  U instance2_;
};

template <typename T>
Abcdefghijklmnopqrstuvwxyz<T, T> construct_obj(T instance) {
  return Abcdefghijklmnopqrstuvwxyz<T, T>(instance, instance);
}
// hand coded
Abcdefghijklmnopqrstuvwxyz<int, int> obj = construct_obj<int>(2);
// using auto
auto obj1 = construct_obj<int>(2);

// keep in mind of perf
std::vector<int> int_values = {1, 2, 3, 4};
// The following code deep-copies int_values into copy_int_values,
// since auto infers the type as std::vector<int>, not std::vector<int>&.
auto copy_int_values = int_values;
// write this instead
auto& ref_int_values = int_values;

// very useful when iterating on containers
std::unordered_map<std::string, int> map;
map.insert({{"andy", 445}, {"jignesh", 645}});
for (auto it = map.begin(); it != map.end(); ++it) {
  std::cout << "(" << it->first << "," << it->second << ")"
            << " ";
}

std::vector<int> vec = {1, 2, 3, 4};
for (const auto& elem : vec) {
  std::cout << elem << " ";
}
```