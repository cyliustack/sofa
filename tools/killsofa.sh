#!/bin/bash
ps aux | grep sofa | awk {'print $2'} | xargs kill -9
