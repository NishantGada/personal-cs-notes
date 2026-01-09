public class Stack {
    private int[] stack;
    public int size = 5;
    private int ptr;

    public Stack() {
        stack = new int[5];
        ptr = 0;
    }
    public Stack(int size) {
        stack = new int[size];
        ptr = 0;
    }

    public boolean isEmpty() {
        if (ptr == 0) {
            return true;
        }
        else return false;
    }
    
    public boolean isFull() {
        if (ptr == stack.length) {
            return true;
        }
        else return false;
    }

    public void display() {
        if (isEmpty()) {
            System.out.println("Stack is Empty!");
            return;
        }

        for (int i=ptr - 1; i > -1; i--) {
            System.out.print(stack[i] + " ");
        }
        System.out.println();
    }

    public void addElement(int data) {
        if (isFull()) {
            System.out.println("Stack is full!");
            return;
        }

        stack[ptr++] = data;
    }

    public int pop() {
        if (isEmpty()) {
            System.out.println("Stack is Empty");
            return -1;
        }
        
        int poppedElement = stack[ptr-1];
        ptr--;
        return poppedElement;
    }
    
    public void peek() {
        if (isEmpty()) {
            System.out.println("Stack is Empty");
            return;
        }
        
        System.out.println(stack[ptr-1]);
    }
}
