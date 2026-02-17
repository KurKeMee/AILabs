# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 08:13:49 2026

@author: Victus
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Функция для расчёта с использованием аггрегирующей функции по столбцу
def best_agg_report (query_start, group_by, group_column, agg, rename, asc):
    return (df.query(query_start)
            .groupby(group_by)
            .agg({group_column:agg})
            .rename(columns={group_column:rename})
            .sort_values(ascending=asc,by=rename))


#Функция для объединения результатов по id
def merge_results (first_serie, second_serie, on_id, asc):
    return (first_serie.merge(second_serie, on=on_id)
                             .sort_values(ascending=asc,
                                          by=[first_serie.columns[0],
                                              second_serie.columns[0]]))

#Считывание файла
df = pd.read_csv("C:/Users/Victus/Desktop/taxi_peru.csv",
                 sep=";",
                 parse_dates=['start_at', 'end_at', 'arrived_at'])
print(df)


#Расчёт количества поездок клиента
rider_count = best_agg_report("end_state == 'drop off'",
                              "user_id",
                              'rider_score',
                              "count",
                              'Количество поездок',
                              False)

print(f"Общее число пользователей {pd.unique(df['user_id']).size}")
print(rider_count)
print(rider_count.describe())

#Расчёт средней оценки клиента водителем
mean_score = best_agg_report("rider_score >= 0 and end_state == 'drop off'",
                              "user_id",
                              "rider_score",
                              "mean",
                              'Средняя оценка',
                              False)
              
print(mean_score)
print(mean_score.describe())

#Обхединение результатов для клиентов
main_stats_rider = merge_results(rider_count,
                                  mean_score,
                                  'user_id',
                                  False)

print(main_stats_rider)


#График для первой оценки лучших клиентов
#ax = sns.barplot(x="Количество поездок", y="Средняя оценка",data = main_stats_rider)
#plt.show()


#Отбор лучших клиентов по количеству поездок (выше 75го перцентиля)
rider_count_75_perc = (rider_count
                       .query("`Количество поездок` >="\
                        f"{rider_count['Количество поездок'].quantile(0.75)}"))

#Отбор лучших клиентов по средней оценке (выше 25го перцентиля)
mean_score_abv_med = mean_score.query("`Средняя оценка` >= "\
                                    f"{mean_score['Средняя оценка'].quantile(0.25)}")

#Объединение результатов
main_stats_rider_new = merge_results(rider_count_75_perc,
                                     mean_score_abv_med,
                                     'user_id',
                                     False)

#График для второй оценки лучших клиентов
#ax = sns.barplot(x="Количество поездок", y="Средняя оценка",data = main_stats_rider_new)
#plt.show()
print("Лучшие пользователи с кол-вом поездок выше 75% и средней оценкой выше 25%")
print(rider_count_75_perc.describe())


#Отбор лучших клиентов по количеству поездок (выше 75го перцентиля)
rider_count_abv_75 = (rider_count_75_perc.query(
                     "`Количество поездок` >= "\
                     f"{rider_count_75_perc['Количество поездок'].quantile(0.75)}"))

#Объединение результатов
main_stats_rider_abv_75 = merge_results(rider_count_abv_75,
                                        mean_score_abv_med,
                                        'user_id',
                                        False)

print(main_stats_rider_abv_75.describe())
#График для оценки лучших клиентов
#ax = sns.barplot(x="Количество поездок", y="Средняя оценка",data = main_stats_rider_abv_75)
#plt.show()

#Отбор лучших клиентов по количеству поездок (выше 75го перцентиля)
rider_count_final = (rider_count_abv_75.query(
                     "`Количество поездок` >= "\
                     f"{rider_count_abv_75['Количество поездок'].quantile(0.75)}"))

#Объединение результатов
main_stats_rider_final = merge_results(rider_count_final,
                                        mean_score_abv_med,
                                        'user_id',
                                        False)

print(main_stats_rider_final.describe())
#График для финальной оценки лучших клиентов
#ax = sns.barplot(x="Количество поездок", y="Средняя оценка",data = main_stats_rider_final)
#plt.show()

#Расчёт количества поездок водителя
driver_count = best_agg_report("end_state == 'drop off'",
                              "driver_id",
                              'driver_score',
                              "count",
                              'Количество поездок',
                              False)
print(f"Общее число водителей {pd.unique(df['driver_id']).size}")
print(driver_count)
print(driver_count.describe())

#Расчёт средней оценки водителя пассажиром
mean_score = best_agg_report("driver_score >= 0 and end_state == 'drop off'",
                              "driver_id",
                              "driver_score",
                              "mean",
                              'Средняя оценка',
                              False)
              
print(mean_score)
print(mean_score.describe())

#Обхединение результатов для водителя
main_stats_driver = merge_results(driver_count,
                                  mean_score,
                                  'driver_id',
                                  False)

print(main_stats_driver)


#График для первой оценки лучших водителей
#ax = sns.barplot(x="Количество поездок", y="Средняя оценка",data = main_stats_driver)
#plt.show()


#Отбор лучших водителей по количеству поездок (выше 75го перцентиля)
driver_count_75_perc = (driver_count
                       .query("`Количество поездок` >="\
                        f"{driver_count['Количество поездок'].quantile(0.75)}"))

#Отбор лучших водителей по средней оценке (выше медианы)
mean_score_abv_med = mean_score.query("`Средняя оценка` >= "\
                                    f"{mean_score['Средняя оценка'].median()}")

#Объединение результатов
main_stats_driver_new = merge_results(driver_count_75_perc,
                                     mean_score_abv_med,
                                     'driver_id',
                                     False)

#График для второй оценки лучших водителей
#ax = sns.barplot(x="Количество поездок", y="Средняя оценка",data = main_stats_driver_new)
#plt.show()

print(main_stats_driver_new.describe())


#Отбор лучших водителей по количеству поездок (выше медианы)
driver_count_final = (driver_count_75_perc.query(
                     "`Количество поездок` >= "\
                     f"{driver_count_75_perc['Количество поездок'].median()}"))

#Объединение результатов
main_stats_driver_final = merge_results(driver_count_final,
                                        mean_score_abv_med,
                                        'driver_id',
                                        False)

print(main_stats_driver_final.describe())
#График для финальной оценки лучших водителей
#ax = sns.barplot(x="Количество поездок", y="Средняя оценка",data = main_stats_driver_final)
#plt.show()

#Выявление популярной платформы по заказам такси
popular_source = (df.groupby("source")
                   .agg({'journey_id':'count'})
                   .sort_values(ascending=False, by="journey_id")
                   .rename(columns={"journey_id":"Кол-во заказов"}))
    
#Расчёт отмененных заказов
popular_source["Отмененные"] = (df.query("end_state != 'drop off' and"\
                                         " end_state.notna()")
                                .groupby("source")
                                .agg({'end_state':'count'})
                                .sort_values(ascending=False, by="end_state")
                                .rename(columns={"journey_id":"Кол-во заказов"}))

print(popular_source)

#Расчёт средней оценки по типам заказа
print(df.query("driver_score>=0")
      .groupby("start_type")
      .agg({'driver_score':'mean'})
      .sort_values(ascending=False, by="driver_score")
      .rename(columns={"driver_score":"Средняя оценка"}))


#Отбор строк с выполненными и оцененными заказами
journey_time = (df.query("end_state == 'drop off' and rider_score >=0")
                .set_index("journey_id")
                .rename(columns={"rider_score":"Оценка пассажира водителем"}))

#Расчёт времени поездки
journey_time["Время поездки в минутах"] = ((journey_time["end_at"] -
                                           journey_time["start_at"])
                                           .dt.total_seconds()//60)

print(journey_time[["Время поездки в минутах", "Оценка пассажира водителем"]])

#Отображение зависимости
ax = sns.barplot(y="Время поездки в минутах", x="Оценка пассажира водителем",data = journey_time)
plt.show()

















