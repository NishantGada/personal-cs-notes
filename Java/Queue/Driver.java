package Queue;

public class Driver {
    public static void main(String[] args) {
        Queue myQueue = new Queue(3);

        myQueue.addElement(1);
        myQueue.display();
        myQueue.addElement(2);
        myQueue.display();
        myQueue.addElement(3);
        myQueue.display();
    }
}
 