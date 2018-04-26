import time
from selenium import webdriver

driver = webdriver.Firefox()
driver.maximize_window()
driver.get("https://datachart.500.com/ssq/history/history.shtml")

driver.find_elements_by_xpath("//input[@id='start']")
# driver.find_elements_by_xpath("//input[@id='start']").send_keys("1")
# # print(value)
time.sleep(10)
driver.close()
