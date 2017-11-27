cpp_file = open("log_cpp.txt")
stl_file = open("log_stl.txt")

def diff(old, new):
    return old*100/new - 100

old = map(float, cpp_file.readline()[:-1].split(" "))
new = map(float, stl_file.readline()[:-1].split(" "))

print
print "Execution time changes (how much faster the STL implementation is):"
print "Acquisition: %.2f%%" % diff(old[0], new[0])
print "Preprocess:  %.2f%%" % diff(old[1], new[1])
print "Track:       %.2f%%" % diff(old[2], new[2])
print "Integrate:   %.2f%%" % diff(old[3], new[3])
print "Render:      %.2f%%" % diff(old[4], new[4])
print
print "Computation: %.2f%%" % diff(old[5], new[5])
print "Total:       %.2f%%" % diff(old[6], new[6])