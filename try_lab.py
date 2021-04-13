import multiprocessing

def print_records(dictionary,records):
    for record in records:
        print("Name: {0}\nScore: {1}\n".format(record[0], record[1]))
        print(dictionary)
    

def insert_record(dictionary,record, records):
    records.append(record)
    print(dictionary)
    print("New record added!\n")
    

if __name__ == '__main__':
    with multiprocessing.Manager() as manager:
        # creating a list in server process memory
        records = manager.list([('Sam', 10), ('Adam', 9), ('Kevin',9)])
        # new record to be inserted in records
        new_record = ('Jeff', 8)

        dictionary = {}
        dictionary['a'] = 'aa'
        dictionary['b'] = 'bb'
        # creating new processes
        p1 = multiprocessing.Process(target=insert_record, args=(dictionary,new_record, records))
        p2 = multiprocessing.Process(target=print_records, args=(dictionary,records,))

        # running process p1 to insert new record
        p1.start()
        p1.join()

        # running process p2 to print records
        p2.start()
        p2.join()
        for i in range(1):
            print(i)
