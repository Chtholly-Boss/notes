# Misc
## wrapper class
**RAII**: Resource Acquisition Is Initialization
```cpp
class IntPtrManager {
  public:
    IntPtrManager() {
      // everything newed should be deleted when unused
      ptr_ = new int;
      *ptr_ = 0;
    }

    // Another constructor for this wrapper class that takes a initial value.
    IntPtrManager(int val) {
      ptr_ = new int;
      *ptr_ = val;
    }

    // destructor
    ~IntPtrManager() {
      if (ptr_) {
        delete ptr_;
      }
    }

    // move constructor and assignment
    IntPtrManager(IntPtrManager&& other) {
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }

    IntPtrManager &operator=(IntPtrManager &&other) {
      if (ptr_ == other.ptr_) {
        return *this;
      }
      if (ptr_) {
        delete ptr_;
      }
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
      return *this;
    }

    IntPtrManager(const IntPtrManager &) = delete;
    IntPtrManager &operator=(const IntPtrManager &) = delete;

    void SetVal(int val) {
      *ptr_ = val;
    }
    int GetVal() const {
      return *ptr_;
    }
  private:
    int *ptr_;

}
```
## iterator
```cpp
// C++ iterators are objects that point to an element inside a container.
// The main components of a C++ iterator are its main two operators.
// 1. dereference operator (*) on an iterator should return the value of the
// element at the current position of the iterator. 
// 2. The ++ (postfix increment) operator should increment 
// the iterator's position by 1. 

// a basic doubly linked list iterator
// In C++, struct and class are essentially the same, 
// with one key difference: the default access specifier.
// * Members of a struct are public by default.
// * Members of a class are private by default.
struct Node {
  Node(int val) 
    : next_(nullptr)
    , prev_(nullptr)
    , value_(val) {}

  Node* next_;
  Node* prev_;
  int value_;
};

class DLLIterator {
  public:
    DLLIterator(Node* head) 
      : curr_(head) {}
    // Implementing a prefix increment operator (++iter).
    DLLIterator& operator++() {
      curr_ = curr_->next_;
      return *this;
    }

    // Implementing a postfix increment operator (iter++).
    DLLIterator operator++(int) {
      DLLIterator temp = *this;
      ++*this;
      return temp;
    }

    bool operator==(const DLLIterator &itr) const {
      return itr.curr_ == this->curr_;
    }

    bool operator!=(const DLLIterator &itr) const {
      return itr.curr_ != this->curr_;
    }

    // dereference operator
    int operator*() {
      return curr_->value_;
    }

  private:
    Node* curr_;
};

// doubly linked list
class DLL {
  public:
    // DLL class constructor.
    DLL() 
    : head_(nullptr)
    , size_(0) {}

    // Destructor should delete all the nodes by iterating through them.
    ~DLL() {
      Node *current = head_;
      while(current != nullptr) {
        Node *next = current->next_;
        delete current;
        current = next;
      }
      head_ = nullptr;
    }

    // Function for inserting val at the head of the DLL.
    void InsertAtHead(int val) {
      Node *new_node = new Node(val);
      new_node->next_ = head_;

      if (head_ != nullptr) {
        head_->prev_ = new_node;
      }

      head_ = new_node;
      size_ += 1;
    }

    // The Begin() function returns an iterator to the head of the DLL,
    // which is the first element to access when iterating through.
    DLLIterator Begin() {
      return DLLIterator(head_);
    }

    // The End() function returns an iterator that marks the one-past-the-last
    // element of the iterator. In this case, this would be an iterator with
    // its current pointer set to nullptr.
    DLLIterator End() {
      return DLLIterator(nullptr);
    }

    Node* head_{nullptr};
    size_t size_;
};
```
## namespaces

```cpp
// declare a namespace
namespace A {
  void foo(int a) {
    std::cout << "Hello from A::foo: " << a << std::endl;
  }
}

// to call functions in A
A::foo(10);
// or you can bring A into current scope
using namespace A;
foo(10);
// or you can just bring foo
using A::foo;
```