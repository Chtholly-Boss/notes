# Intro
## Database Applications:
* online transaction processing
* data analytics

## Purposes of a Database Management System
* Disadvantage of File-Processing System
  * Data redundancy and inconsistency
  * Difficulty in accessing data
    * Not allow needed data to be retrieved in a convenient and efficient manner
  * Data isolation: data scattered in various files
  * Integrity problems: consistency constraints
    * when a new constraint added, changes to many files should be made
  * Atomicity problems
  * Concurrent-access anomalies
  * Security problems

## View of Data
Data Models:
* Relational Model: Record-Based Model
* Entity-Relationship (E-R) Model: 
* Semi-Structured Data Model
* Object-Based Data Model

Data Abstraction (bottom-up)
* Physical Level: how data are actually stored
* Logical Level: what data are stored and what relationships exist among them
* View Level: simplify user's interaction with the database

```c
// logical level
typedef struct {
  char  id[5];
  char  name[20];
  double  salary;
} instructor;
// physical level: a block of consecutive bytes
// view model: students couldn't see instructor's salary
```

Instance
  : The collection of information stored in the database at a particular moment

Schema
  : The overall design of the database

## Database Languages
* Data Definition Language (DDL): Specify a Database Schema
  * Data Storage and Definition
  * Constraints
    * Domain Constraints
    * Referential Integrity
    * Authorization

```sql
// SQL Data-Definition Language
create table department
  (
    dept_name char(20),
    building  char(15),
    budget    numeric(12,2)
  );
```

* Data Manipulation Language (DML)
  * type of access:
    * Retrieval
    * Insertion
    * Deletion
    * Modification
  * types of DML
    * Procedural DMLs: specify what data needed and how to get those data
    * Declarative DMLs: only specify what data needed

```sql
// SQL Data-Manipulation Language
select instructor.name
from instructor
where instructor.dept_name = 'History';
```
Meta-data
: data about data

Query
: a statement requesting the retrieval of information

## Database Engine
Functional Components:
* Storage Manager
  * Authorization and integrity manager
  * Transaction manager
  * File manager
  * Buffer manager
* Query Processor Components
  * DDL interpreter
  * DML compiler
  * Query evaluation engine
* Transaction Management Component
  * Concurrency-control manager
  * Recovery manager

![dbs-archi](./figure/dbs_arch.png)

* Users:
  * Naive Users
  * Application Programmers
  * Sophisticated Users
  * Administrator
