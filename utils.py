# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:07:20 2015

@author: thalita
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from time import time
import smtplib
import getpass
from email.mime.text import MIMEText


# a fixed random_seed randomly picked once
# used np.random.rand() and picked first 8 decimal digits
RANDOM_SEED = 13986301

def oneD(array):
    # TODO problably the ndarray.flatten method can do this
    return np.array(np.array(array).squeeze(), ndmin=1)

class timing(object):
    def __init__(self):
        self.t0 = 0
        self.tic()
    def tic(self):
        self.t0 = time()
    def toc(self, text=''):
        if text != '':
            text == ' '
        dt = time()-self.t0
        print(text, 'Time elapsed:',dt,' s')
        self.tic()
        return dt

class Notifier(object):
    def __init__(self, opt_msg=''):
        self.FROM = "thalitafdrumond@gmail.com"
        self.TO = self.FROM # must be a list

        SUBJECT = 'Done'

        TEXT = "Finished script\n" + opt_msg

        # Prepare actual message
        msg = MIMEText(TEXT)
        msg['Subject'] = SUBJECT
        msg['From'] = self.FROM
        msg['To'] = self.TO
        self.message = msg.as_string()
        self.server = smtplib.SMTP_SSL('smtp.gmail.com:465')
        self.passwd = None

        for i in range(3):
            try:
                self.passwd = getpass.getpass('%s gmail password:' % self.FROM)
                self.server.login(self.FROM, self.passwd)
                break
            except smtplib.SMTPAuthenticationError:
                if i < 2:
                    print('Wrong password, try again')
                else:
                    print ('Wrong password, exiting')

    def notify(self):
        # Send the mail
        self.server.login(self.FROM, self.passwd)
        self.server.sendmail(self.FROM, self.TO, self.message)
        self.server.quit()


def read_result(fname, path='', meanstd=True):
    splitted = fname[:fname.find('pct')-4].split('_')
    params = {'RStype': splitted[0]}
    for i in range(1, len(splitted)-1, 2):
        value = splitted[i+1]
        try:
            value = float(value)
        except ValueError:
            pass
        params[splitted[i]] = value

    with open(path + fname, 'r') as f:
        header = f.readline()
    header = header.replace('#', '').replace('"', '').replace(' ', '')\
        .replace('\n', '').split(',')
    header = header + [h+'(std)' for h in header]
    result = np.loadtxt(path + fname, delimiter=',', ndmin=2)
    if meanstd:
        result = np.hstack((result.mean(axis=1), result.std(axis=1)))
    result = dict(zip(header, result))
    result.update(params)
    return result


def read_results(path='', meanstd=True):
    fnames = [f for f in os.listdir(path) if f.find('test.txt') > -1]
    result = []
    for fname in fnames:
        result.append(read_result(fname, path, meanstd=meanstd))
    return result


def pd_select(dataframe, select):
    try:
        data = dataframe
        for key, value in select.items():
            data = data[data[key] == value]
            if len(data) == 0:
                return None
        return data
    except KeyError:
        return None
