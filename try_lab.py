first_name = "tung"
last_name = "nguyen"

us_full_name_set = frozenset([first_name,last_name]);
vn_full_name_set = frozenset([last_name,first_name]);
person = {};
person[us_full_name_set] = first_name+"_"+last_name;
person[vn_full_name_set] = last_name + "_" + first_name;

print(len(person));
print(person[us_full_name_set]);
print(person[vn_full_name_set]);