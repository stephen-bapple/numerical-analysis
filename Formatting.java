public class Formatting {
    public static void main(String... args) {
        double num = 1233.245324862;
        String printStr = String.format("%8.2f", num);
        System.out.println(printStr);
    }
    
}