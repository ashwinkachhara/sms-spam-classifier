with open('SMSSpamCollection') as file:
    dataset = [(x.split('\t')[0],x.split('\t')[1]) for x in [line.strip() for line in file]]
