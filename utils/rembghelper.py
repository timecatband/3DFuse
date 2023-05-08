from rembg import remove
import sys
input_path = sys.argv[1]
output_path = sys.argv[2]

with open(input_path, 'rb') as i:
    with open(output_path, 'wb') as o:
        input = i.read()
        output = remove(input)
        o.write(output)
