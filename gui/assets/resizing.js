/*
 * Copyright (c) 2023 Salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 *
 */

/* resize figures in table upon callback get fires */

if(!window.dash_clientside) {window.dash_clientside = {};}
window.dash_clientside.clientside = {
   resize: function (value) {
       console.log("resizing...");
       window.dispatchEvent(new Event('resize'));
       return null
   }
}