import configparser

class Properties:
    def __init__(self, property_file_name="Properties/properties.ini"):
        self.property_file_name = property_file_name
        self.property_file = configparser.ConfigParser()
        self.property_file.read(self.property_file_name)

    def get_training_data_path(self):
        return self.property_file['DEFAULT']['train_data']

    def get_items_data_path(self):
        return self.property_file['DEFAULT']['items_data']

    def get_train_dataset_path(self):
        return self.property_file['EVALUATION']['train_module']

    def get_test_dataset_path(self):
        return self.property_file['EVALUATION']['test_module']




    def get_training_data_file(self):
        return open(self.get_training_data_path(), "rb")

    def get_items_data_file(self):
        return open(self.get_items_data_path(), "rb")

    def get_train_dataset_file(self):
        return open(self.get_train_dataset_path(), "rb")

    def get_test_dataset_file(self):
        return open(self.get_test_dataset_path(), "rb")





    def get_epocs(self):
        return self.property_file['ALGORITHM']['epochs']


    def get_actor_learning_rate(self):
        return self.property_file['ALGORITHM']['actor_learning_rate']


    def get_critic_learning_rate(self):
        return self.property_file['ALGORITHM']['critic_learning_rate']



    def get_property_file(self):
        return self.property_file
