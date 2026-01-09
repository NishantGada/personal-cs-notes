class Driver {
    public static void main(String[] args) {
        Stack myStack = new Stack(3);

        myStack.addElement(11);
        myStack.addElement(22);
        myStack.addElement(33);
        myStack.display();

        myStack.peek();
        myStack.display();

        myStack.pop();
        myStack.display();

        myStack.pop();
        myStack.display();
    }
}
