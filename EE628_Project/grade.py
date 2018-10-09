#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 01:52:32 2018

@author: bingyangwen
"""
a= []
while True:
    student_key = ['Name', 'Total Grades', 'Q1 Grades', 'Q2 Grades', 'Q3 Grades', 'Q5 Grades', 'Q6 Grades']
    
    name = raw_input('Enter name:')
    if name == 'exit':
        break
    Q1 = 20-float(raw_input('Q1 deduction:'))
    Q2 = 20-float(raw_input('Q2 deduction:'))
    Q3 = 20-float(raw_input('Q3 deduction:'))
    Q5 = 20-float(raw_input('Q5 deduction:'))
    Q6 = 20-float(raw_input('Q6 deduction:'))
    Tot_grade = Q1+Q2+Q3+Q5+Q6
    infor = [name, Tot_grade, Q1, Q2, Q3, Q5, Q6]
    infor = zip(student_key,infor)
    a.append(infor)
    