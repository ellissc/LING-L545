import sys
import matplotlib.pyplot as plt

vo = 0
ov = 0

'''
#prints out the order for a given conllu
for c in sys.stdin.readlines():
	if c[0] != "#":
		#c = c[c.find("_")+2:]
		#c = c[:c.find("_")-1]
		if "\tobj\t" in c and "root|" not in c:
			#print(c)
			index = int(c[0:2]) #of the object
			#print(index)
			head = int(c[c.find("\tobj\t")-2:c.find("\tobj\t")])
			#print(head)
			if index > head: #root object, verb object
				vo +=1
			elif index <= head:
				ov +=1
			#print()
			#print()
print("verb object: ", vo)
print("object verb: ", ov)
'''

'''
	English:
verb object:  9804, 0.96
object verb:  377, 0.04
10181
	Italian:
verb object:  8371, 0.88
object verb:  1118, 0.12
9489
	korean:
verb object: 125, 0.03
object verb: 3997, 0.97
4122
	Portuguese
verb object: 8050, 0.91
object verb: 775, 0.09
8825
	turkish:
verb object: 136, 0.05
object verb: 2632, 0.95
2768
	Chinese:
verb object: 6157, 0.997
object verb: 18, 0.003
6175
	Hebrew:
verb object: 3224, 0.96
object verb: 119, 0.04
3343
	japanese:
verb object: 1, 0.0003
object verb: 4425, 0.9997
4426
	Norwegian:
verb object: 10277, 0.94
object verb: 594, 0.06
10871
	Russian:
verb object: 21309, 0.80
object verb: 5399, 0.2
26708
'''
labels = {0:'eng', 1:'ita', 2:'kor',3:'por',4:'tur',5:"chi",6:"heb",7:"jap",8:"nor",9:"rus"}
x = [0.04,0.12,0.97,0.09,0.95,0.003,0.04,0.9997,0.06,0.2]  # proportion of OV
y = [0.96,0.88,0.03,0.91,0.05,0.997,0.96,0.0003,0.94,0.8]  # proportion of VO
plt.plot(x, y, 'ro')
plt.title('Relative word order of verb and object')
plt.xlim([0,1]) # Set the x and y axis ranges
plt.ylim([0,1])
plt.xlabel('OV') # Set the x and y axis labels
plt.ylabel('VO')
for i in labels:  # Add labels to each of the points
    plt.text(x[i]-0.03, y[i]-0.03, labels[i], fontsize=9)
plt.savefig("verb_object_order.png")
plt.show()