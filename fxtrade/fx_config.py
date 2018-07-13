import configparser as cp

# ToDo : もうちょっとPythonらしく書く

class FxConfig:

    __section = 'FX_SETTINGS'
    __config_path = "/Users/kitano.kenta/PycharmProjects/fxtrade/fx_config.ini"

    __config = cp.ConfigParser()


    def __init__(self):
        try:
            self.__config.read(self.__config_path)
        except IOError as e:
            print("Config file \"{0}\" is not found.".format(e))


    def get_model_save_filename(self):
        try:
            path = self.__config.get(self.__section, "model_save_path")
            filename = self.__config.get(self.__section, "model_save_filename")
            ret = path + filename
        except cp.NoOptionError as e:
            print("ERROR : {}".format(e))
            ret = None

        return ret


    def get_scale_save_filename(self):
        try:
            path = self.__config.get(self.__section, "scale_save_path")
            filename = self.__config.get(self.__section, "scale_save_filename")
            ret = path + filename
        except cp.NoOptionError as e:
            print("ERROR : {}".format(e))
            ret = None

        return ret


    def get_rate_filename(self):
        try:
            path = self.__config.get(self.__section, "rate_file_path")
            filename = self.__config.get(self.__section, "rate_file_name")
            ret = path + filename
        except cp.NoOptionError as e:
            print("ERROR : {}".format(e))
            ret = None

        return ret


    def get_rate_filename_for_debug(self):
        try:
            path = self.__config.get(self.__section, "rate_file_path_debug")
            filename = self.__config.get(self.__section, "rate_file_name_debug")
            ret = path + filename
        except cp.NoOptionError as e:
            print("ERROR : {}".format(e))
            ret = None

        return ret


    def get_target_pair(self):

        dic = {"USDJPY" : 1, "EURJPY" : 2}

        try:
            target_pair = self.__config.get(self.__section, "target_pair")
            items = target_pair.split(',')
            ret = []
            for x in items:
                if x in dic:
                    ret.append(dic[x])
                else:
                    print("ERROR : key error in target pair")
        except cp.NoOptionError as e:
            print("ERROR : {}".format(e))
            ret = None

        return ret


    def __get_int_value(self, option):
        try:
            val = self.__config.get(self.__section, option)
            ret = int(val)
        except cp.NoOptionError as e:
            print("ERROR : {}".format(e))
            ret = None
        return ret


    def __get_bool_value(self, option):
        flag = 'True'
        ret = True
        try:
            flag = self.__config.get(self.__section, option)
        except cp.NoOptionError as e:
            print("ERROR : {}".format(e))
            ret = None

        if flag == 'False':
            ret = False

        return ret


    def __get_str(self, option):
        try:
            str = self.__config.get(self.__section, option)
        except cp.NoOptionError as e:
            print("ERROR : {}".format(e))
            str = None

        return str



    def get_model_save_flag(self):
        return self.__get_bool_value("model_save")


    def get_batch_size(self):
        return self.__get_int_value("batch_size")


    def get_sequence_len(self):
        return self.__get_int_value("sequence_len")


    def get_predict_len(self):
        return self.__get_int_value("predict_len")


    def get_hidden_num(self):
        return self.__get_int_value("hidden_num")


    def get_output_currency_flag(self):
        return self.__get_bool_value("add_info_currency")


    def get_output_hour_flag(self):
        return self.__get_bool_value("add_info_hour")


    def get_output_minute_flag(self):
        return self.__get_bool_value("add_info_minute")


    def get_output_second_flag(self):
        return self.__get_bool_value("add_info_second")


    def get_output_weekday_flag(self):
        return self.__get_bool_value("add_info_weekday")


    def get_scale_each_sequence_flag(self):
        return self.__get_bool_value("scale_each_sequence")


    def get_scale_all_sequence_flag(self):
        return self.__get_bool_value("scale_all_sequence")


    def get_scale_save_flag(self):
        return self.__get_bool_value("scale_save")


    def get_class_method(self):
        return self.__get_str("class_method")


    def get_model_name(self):
        return self.__get_str("model")


    def get_plot_history_flag(self):
        return self.__get_bool_value("plot_history")


    def get_show_model_summary_flag(self):
        return self.__get_bool_value("show_model_summary")


# テストコード
if __name__ == '__main__':

    config = FxConfig()

    print(config.get_model_save_filename())
    print(config.get_model_save_flag())
    print(config.get_rate_filename_for_debug())
    print(config.get_rate_filename())
    print(config.get_target_pair())
    print(config.get_batch_size())
    print(config.get_predict_len())
    print(config.get_sequence_len())
    print(config.get_output_currency_flag())
    print(config.get_output_hour_flag())
    print(config.get_output_minute_flag())
    print(config.get_output_second_flag())
    print(config.get_output_weekday_flag())