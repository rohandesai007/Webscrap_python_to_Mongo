import datetime


def pass_date_param():
    today_date = datetime.date.today()
    yesterday_date = today_date - datetime.timedelta(days=1)
    formatted_date = datetime.date.strftime(yesterday_date, "X%m/X%d/%y").replace('X0', 'X').replace('X', '')
    return str(formatted_date)


pattern = pass_date_param()
load_case_data(url_death, mongo_url, 'deaths_cases', pattern)
# load_case_data(url_confirmed, mongo_url, 'confirmed_cases', pattern)
# load_case_data(url_recovered, mongo_url, 'recovered_cases', pattern)
# Data_prep_for_ml(mongo_url)
data_for_dc = data_prep_death_cases(mongo_url)
var2 = plot_death_over_time(data_for_dc)

