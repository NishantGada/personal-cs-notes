package Queue;

public class Queue {
    protected int[] queue;
    protected int start;
    protected int end;
    protected int size;

    public Queue() {
        this.queue = new int[3];
        this.start = 0;
        this.end = 0;
    }
    public Queue(int size) {
        this.queue = new int[size];
        this.start = 0;
        this.end = 0;
    }

    public boolean isEmpty() {
        if (start == end) {
            return true;
        }
        else return false;
    }
    
    public boolean isFull() {
        if (end == queue.length) {
            return true;
        }
        else return false;
    }

    public void addElement(int data) {
        if (isFull()) {
            System.out.println("Queue is Full!");
            return;
        }

        queue[end] = data;
        end++;
    }

    public void display() {
        if (isEmpty()) {
            System.out.println("Queue is Empty!");
            return;
        }

        for (int i=start; i<=end - 1; i++) {
            System.out.print(queue[i] + " ");
        }
        System.out.println();
    }
}
