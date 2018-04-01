public class Ex {
    public static void main(String args[]) {
     
        
        String[][] array = {{"10L", "45F", "53A", "31A"},
                    {"10L", "45F", "53A", "31A"},
                    {"10L", "45F", "53A", "31A"},
                    {"10L", "45F", "53A", "31A"}};
        
        
        String fs = "";
        
        final int padSize = 4;

        int size = (padSize + 1) * array[0].length + 1;
        
        
        for (int i = 0; i < size; i++) {
            if (i % (padSize + 1) == 0) {
                fs += "+";
            }
            else {
                fs += "-";
            }
        }

        System.out.println(fs);
        
        for (String[] row : array) {
            fs = "|";
            for (String entry : row) {
                fs = fs + "%" + padSize + "s|";
            }
            fs += "\n";
             
            for (int i = 0; i < size; i++) {
                if (i % (padSize + 1) == 0) {
                    fs += "+";
                }
                else {
                    fs += "-";
                }
            }
            
            System.out.printf(fs + "\n", row);
            
        }
    }
}