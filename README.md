# polyominos_tiling_ga
Проблема. Для данного размера прямоугольника-стола T и данного множества опорных прямоуголных полиомино и опорных L-полиомино Pi с данными соответствующими мощностями Ni узнать, существует ли конфигурация полиомино с этими параметрами, являющееся замощением T.
Реализован генетический алгоритм с помощью deap. Основная идея: рандомим положение полимино и если они пересекаются назначается штраф и алгоритм пытается его минимизировать, если целевая функция равна нулю - размещение удалось, в ином случае нет
Попытался сделать красивую визуализацию, но не доделал её. В остальном алгоритм работает прекрасно. Нужно скачать ga.py и algelitism.py установить deap, numpy запустить файл ga.py 

Пусть K - число этапов эволюции, реализуемых ГА, M - число особей в популяции, W - объем памяти (в битах) для хранения одной особи, a F - множество рассматриваемых неисправностей.

Оценка алгоритмической сложности одного запуска генетического алгоритма равна $$O(KM\cdot(W\cdot(|F|+1)+\log{M}))$$, а объем памяти, необходимой для работы генетического алгоритма, равен $$O(2MW)$$.
