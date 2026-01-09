# Factory Method design pattern

In this design pattern, we somewhat outsource the creation of an object to new class, which will handle object creation for us - based on the type of object that we request from it in our code. 

For example:
- If you have these concrete classes - Dog, Cat, Bird
- In your code, if you need to interchangeably use objects of these classes, you will have to keep creating/using objects of them everywhere. 
- to solve this problem, you can instead create a new "Factory" class - called AnimalFactory - which will have a method called say "createAnimal"
- this "createAnimal" method would be static, so that you don't have to create an object this FactoryClass too and instead directly access it
- Internally, this "createAnimal" method would be creating specific objects based on the type that we give it in the form of a parameter

- something like this:
```java
// The Factory
class AnimalFactory {
    
    // Constants (final) for animal types
    public static final String DOG = "dog";
    public static final String CAT = "cat";
    
    // Static method to create animals
    public static Animal createAnimal(String type) {
        if (type.equals(DOG)) {
            return new Dog();
        } 
        else if (type.equals(CAT)) {
            return new Cat();
        }
        return null;
    }
}


// Usage
public class Main {
    public static void main(String[] args) {
        // Ask factory for animals
        Animal pet1 = AnimalFactory.createAnimal(AnimalFactory.DOG);
        Animal pet2 = AnimalFactory.createAnimal(AnimalFactory.CAT);
        
        // Use them
        pet1.makeSound();  // Woof!
        pet1.eat();        // Dog eats bones
        
        pet2.makeSound();  // Meow!
        pet2.eat();        // Cat eats fish
    }
}
```

- The variables "DOG" and "CAT" are defined as final so that we can avoid and catch any typos that might occur when requesting the object. 


### where's the interface?

- In our example, the interface is the Animal interface (not the factory itself - this was my confusion originally)
- We create the Animal interface instead of a concrete class so that any common behavior between related objects/classes (such as a sound made by Dog, Cat, Bird, etc) can be grouped and defined together as a contract.


- Here's an example of what using an Interface would like:
```java
// The contract
interface Animal {
    void makeSound();
    void eat();
}

// Different animals
class Dog implements Animal {
    @Override
    public void makeSound() {
        System.out.println("Woof!");
    }
    
    @Override
    public void eat() {
        System.out.println("Dog eats bones");
    }
}

class Cat implements Animal {
    @Override
    public void makeSound() {
        System.out.println("Meow!");
    }
    
    @Override
    public void eat() {
        System.out.println("Cat eats fish");
    }
}
```


- Without interface, our code would look like this:
```java
// If we didn't use interface, we'd have to do this:
Dog pet1 = AnimalFactory.createAnimal(AnimalFactory.DOG);  // Returns Dog
Cat pet2 = AnimalFactory.createAnimal(AnimalFactory.CAT);  // Returns Cat

pet1.makeSound();  // Works
pet2.makeSound();  // Works
```

- Without using an Interface, our factory code would have to deal with every new class (Animal Type) separately, like this:
```java
public void playWithPet(String petType) {
    if (petType.equals("dog")) {
        Dog pet = new Dog();
        pet.makeSound();
        pet.eat();
    } 
    else if (petType.equals("cat")) {
        Cat pet = new Cat();
        pet.makeSound();
        pet.eat();
    }
    // Repeat for every animal type! 😱
}
```