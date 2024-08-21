# The Lox Language
## Basic Syntax
### Data Types
* Booleans: `true` or `false`
* Numbers: only doulbe-precision floating decimal literals
* Strings
* Nil: `nil`

Expressions:
* arithmetic: `+ - * / `
* comparison and equality: `>, >=, <, <=, ==, !=`
* logic: `! and or`

### Statements
```C
// expression statement
"some expression";`
// block 
{
    expression 1;
    expression 2;
    ...
}
```

### Variables: `var`, default to `nil` if not initialized

### Control Flow
```C
// if-statement
if (condition) {
  print "yes";
} else {
  print "no";
}
// while loop
var a = 1;
while (a < 10) {
  print a;
  a = a + 1;
}
// for loop
for (var a = 1; a < 10; a = a + 1) {
  print a;
}
```

### Functions
```C
// declaration
fun printSum(a, b) {
  print a + b;
  return a + b; // implicitly Nil if not return anything
}
// calling
printSum(1,2);
```

### Closures
```C
// pass a function to a function
fun addPair(a, b) {
  return a + b;
}

fun identity(a) {
  return a;
}

print identity(addPair)(1, 2); // Prints "3".

// local function
fun outerFunction() {
  fun localFunction() {
    print "I'm local!";
  }

  localFunction();
}

// return function
fun returnFunction() {
  var outside = "outside";

  fun inner() {
    print outside;
  }

  return inner;
}

var fn = returnFunction();
fn();
```

### Class
```java
class Breakfast {
// method declared without fun keyword
  cook() {
    print "Eggs a-fryin'!";
  }

  serve(who) {
    print "Enjoy your breakfast, " + who + ".";
  }
}

// Store it in variables.
var someVariable = Breakfast;

// Pass it to functions.
someFunction(Breakfast);

var breakfast = Breakfast();
print breakfast; // "Breakfast instance".

// Assigning to a field, creates it if it doesn’t already exist.
breakfast.meat = "sausage";
breakfast.bread = "sourdough";

class Breakfast {
  init(meat, bread) {
    this.meat = meat;
    this.bread = bread;
  }

  // ...
}

var baconAndToast = Breakfast("bacon", "toast");
baconAndToast.serve("Dear Reader");
// "Enjoy your bacon and toast, Dear Reader."

// Brunch inherit Breakfast
class Brunch < Breakfast {
  init(meat, bread, drink) {
    super.init(meat, bread);
    this.drink = drink;
  }
}
```
