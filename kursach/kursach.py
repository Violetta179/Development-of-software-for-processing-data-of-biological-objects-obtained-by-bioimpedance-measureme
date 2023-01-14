
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from scipy import stats
from sklearn.metrics import r2_score

#загрузка данных

df = pd.read_excel("D:\Download/biodata.xlsx", sheet_name ="Лист1")

#выбор данных
xdata = df["Весс"]
ydata = df["Окружность талии"]


#Нарисовать линию графика
curvex = np.linspace(xdata.min(),xdata.max(),1000)

#показательная регрессия1
def power_low(x, a1, b1):
    return a1*b1**x

popt, pcov = scipy.optimize.curve_fit(power_low, xdata, ydata)
a1, b1 = popt

curpow =power_low(curvex, a1, b1)

#Детерминант
squaredDiffs1 = np.square(ydata - power_low(xdata, a1, b1))
squaredDiffsFromMean1 = np.square(ydata - np.mean(ydata))
rSquared1 = 1 - np.sum(squaredDiffs1) / np.sum(squaredDiffsFromMean1)

#степенная регресся2
def power_law(x, a2, b2):
    return a2*x**b2

popt, pcov = scipy.optimize.curve_fit(power_law, xdata, ydata)
a2, b2 = popt
curpaw =power_law(curvex, a2, b2)

# determine
squaredDiffs2 = np.square(ydata - power_law(xdata,a2, b2))
squaredDiffsFromMean2 = np.square(ydata - np.mean(ydata))
rSquared2 = 1 - np.sum(squaredDiffs2) / np.sum(squaredDiffsFromMean2)


#полимеальная  регрессия3
mymodel = np.poly1d(np.polyfit(xdata, ydata, 3))

#Linear regrassion4
slope, intercept, r, p, std_err = stats.linregress(xdata, ydata)
def myfunc(x):
    return slope * x + intercept

mymodel1 = list(map(myfunc, curvex))

squaredDiffs4 = np.square(ydata - myfunc(xdata))
squaredDiffsFromMean4 = np.square(ydata - np.mean(ydata))
rSquared4 = 1 - np.sum(squaredDiffs4) / np.sum(squaredDiffsFromMean4)


#sin regrassion5
def sinfunc(x, a5, b5, c5, d5):
    return a5 * np.sin(b5 * (x - np.radians(c5)))+d5

popt, pcov = scipy.optimize.curve_fit(sinfunc, xdata, ydata)
a5, b5, c5, d5 = popt
cursin = sinfunc(curvex, a5, b5, c5, d5)

squaredDiffs5 = np.square(ydata - sinfunc(xdata, a5, b5, c5, d5))
squaredDiffsFromMean5 = np.square(ydata - np.mean(ydata))
rSquared5 = 1 - np.sum(squaredDiffs5) / np.sum(squaredDiffsFromMean5)


#exp regrassion6
def monoExp(x, a6, b6):
    return np.exp(a6 +b6*x) 

params, cv = scipy.optimize.curve_fit(monoExp, xdata, ydata)
a6, b6  = params
curexp = monoExp(curvex, a6, b6)

# determine 
squaredDiffs6 = np.square(ydata - monoExp(xdata, a6, b6))
squaredDiffsFromMean6 = np.square(ydata - np.mean(ydata))
rSquared6 = 1 - np.sum(squaredDiffs6) / np.sum(squaredDiffsFromMean6)

#gip regrassion7
def gipfunc(x, a7, b7):
    return a7 + b7/ x 

popt, pcov = scipy.optimize.curve_fit(gipfunc, xdata, ydata)
a7, b7= popt
curgip = gipfunc(curvex, a7, b7)
squaredDiffs7 = np.square(ydata - gipfunc(xdata, a7, b7))
squaredDiffsFromMean7 = np.square(ydata - np.mean(ydata))
rSquared7 = 1 - np.sum(squaredDiffs7) / np.sum(squaredDiffsFromMean7)


#log regrassion8
def func(x,a8 ,b8):
  return a8* np.log(x)+b8

popt, pcov = scipy.optimize.curve_fit(func, xdata, ydata)
a8, b8 = popt
curlog = func(curvex, a8 ,b8)

squaredDiffs8 = np.square(ydata - func(xdata, a8 ,b8))
squaredDiffsFromMean8 = np.square(ydata - np.mean(ydata))
rSquared8 = 1 - np.sum(squaredDiffs8) / np.sum(squaredDiffsFromMean8)

figure = plt.figure()
ax1 = figure.add_subplot(4,2,1)
ax1.scatter(xdata,ydata, color = 'r')
ax1.plot(curvex, curpow, '--', linewidth =5)
ax1.legend([f"R²= {rSquared1:.5f}, Y = {a1:.5f} * {b1:.5f}^x"])


ax2 = figure.add_subplot(4,2,2)
ax2.scatter(xdata,ydata, color = 'r')
ax2.plot(curvex, curpaw, color='g')
ax2.legend([f"R²= {rSquared2:.5f},Y = {a2:.5f} * x^{b2:.5f} "] )


ax3 = figure.add_subplot(4,2,3)
ax3.scatter(xdata,ydata, color = 'r')
ax3.plot(curvex, mymodel(curvex))
ax3.legend([f"Y={mymodel},R²= {r2_score(ydata, mymodel(xdata)):.5f}"])


ax4 = figure.add_subplot(4,2,4)
ax4.scatter(xdata,ydata, color = 'r')
ax4.plot(curvex, mymodel1)
ax4.legend([f"R²= {rSquared4:.5f}, Y = {slope:.5f} * x + {intercept:.5f}"])


ax5 = figure.add_subplot(4,2,5)
ax5.scatter(xdata,ydata, color = 'r')
ax5.plot(curvex, cursin)
ax5.legend([f"R²= {rSquared5:.5f},Y = {a5:.5f}*sin({b5:.3f}*(x-{np.radians(c5):.5f}))+{d5:.5f}]"])


ax6 = figure.add_subplot(4,2,6)
ax6.scatter(xdata,ydata, color = 'r')
ax6.plot(curvex, curexp)
ax6.legend([f"R²= {rSquared6:.5f},Y =  e^({a6:.5f} +{b6:.5f}*x)]"])


ax7 = figure.add_subplot(4,2,7)
ax7.scatter(xdata,ydata, color = 'r')
ax7.plot(curvex, curgip)
ax7.legend([f"R²= {rSquared7:.5f},Y = {a7:.5f} + {b7:.5f}/x"])


ax8 = figure.add_subplot(4,2,8)
ax8.scatter(xdata,ydata, color = 'r')
ax8.plot(curvex, curlog)
ax8.legend([f"R²= {rSquared8:.5f},Y = {a8:.5f} + {b8:.5f}*log(x)"])

plt.show()
